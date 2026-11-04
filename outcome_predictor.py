"""Context-aware recovery outcome estimates from deterministic trial history."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt

ALLOWED_ACTIONS = {
    "REPLAN",
    "WAIT",
    "RETRY",
    "GATHER_INFORMATION",
    "REQUEST_HUMAN_HELP",
}


def build_action_statistics(results):
    statistics = defaultdict(lambda: {"success": 0, "failure": 0})

    for trial in results:
        for action, succeeded in getattr(trial, "recovery_outcomes", ()):
            if action in ALLOWED_ACTIONS:
                statistics[action]["success" if succeeded else "failure"] += 1

    return {action: dict(counts) for action, counts in statistics.items()}


def _wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0

    p = successes / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * sqrt(
        (p * (1 - p) + z * z / (4 * total)) / total
    )

    return max(0.0, (centre - margin) / denominator)


def predict_outcome(action, environment_state, statistics):
    """Return a conservative estimate with feasibility and evidence."""

    if action not in ALLOWED_ACTIONS:
        return {
            "success_probability": 0.0,
            "lower_confidence_bound": 0.0,
            "risk_level": "HIGH",
            "expected_result": "Unsupported action.",
            "observations": 0,
            "feasible": False,
        }

    state = environment_state or {}

    feasible = True
    constraint = "No immediate constraint detected."

    # Existing navigation safety constraints.
    if action == "REPLAN" and not state.get("safe_alternative_route", False):
        feasible = False
        constraint = "No safe alternative route is confirmed."

    elif action == "RETRY" and state.get("path_blocked", False):
        feasible = False
        constraint = "The target path is still blocked; retry is unsafe."

    elif action == "WAIT" and not state.get(
        "temporary_obstacle",
        state.get("obstacle_temporary", False),
    ):
        constraint = "No temporary-obstacle evidence; waiting benefit is uncertain."

    elif action == "REQUEST_HUMAN_HELP":
        constraint = "Safe escalation option; it does not autonomously restore progress."

    # Inspect structured visual evidence.
    visual_evidence = state.get("visual_evidence", {})

    if not isinstance(visual_evidence, dict):
        visual_evidence = {}

    evidence_status = visual_evidence.get(
        "evidence_status",
        "NO_VERIFIED_VISUAL_INPUT",
    )

    uncertainties = visual_evidence.get("uncertainties", [])

    if not isinstance(uncertainties, list):
        uncertainties = []

    analysis_completed = evidence_status == "VISUAL_EVIDENCE_ANALYZED"

    visual_warning = ""

    if not analysis_completed:
        visual_warning = " Visual analysis is unavailable or incomplete."

    elif uncertainties:
        visual_warning = (
            f" Visual analysis contains {len(uncertainties)} unresolved uncertainties."
        )

    # Action-specific uncertainty handling.
    uncertainty_reason = ""

    if uncertainties or not analysis_completed:

        if action == "RETRY":
            if not state.get("path_blocked", False):
                feasible = False
                uncertainty_reason = (
                    "Retry requires sufficient environmental evidence "
                    "to confirm that the path is clear."
                )

        elif action == "REPLAN":
            if not state.get("safe_alternative_route", False):
                feasible = False
                uncertainty_reason = (
                    "Replanning requires a confirmed safe alternative route."
                )

        elif action == "WAIT":
            if not state.get(
                "temporary_obstacle",
                state.get("obstacle_temporary", False),
            ):
                uncertainty_reason = (
                    "Waiting may be appropriate, but temporary-obstacle "
                    "evidence is missing."
                )

        elif action == "GATHER_INFORMATION":
            uncertainty_reason = (
                "Gather information to reduce uncertainty before acting."
            )

        elif action == "REQUEST_HUMAN_HELP":
            uncertainty_reason = (
                "Human assistance remains available despite visual uncertainty."
            )

    # Historical statistics remain independent of visual evidence.
    counts = statistics.get(action, {})

    successes = int(counts.get("success", 0))
    failures = int(counts.get("failure", 0))
    total = successes + failures

    if total == 0:
        probability = 0.5
        lower = 0.0
        risk = "UNKNOWN"

        expected = (
            "No historical observations; probability is an uninformative "
            "prior, not evidence."
        )

    else:
        probability = (successes + 1) / (total + 2)
        lower = _wilson_lower_bound(successes, total)

        risk = (
            "LOW" if lower >= 0.60
            else "MEDIUM" if lower >= 0.30
            else "HIGH"
        )

        expected = (
            f"Historical estimate from {total} observations "
            f"({successes} successes, {failures} failures)."
        )

    # Reject actions that violate safety constraints.
    if not feasible:
        risk = "HIGH"
        probability = 0.0
        lower = 0.0

    return {
        "success_probability": round(probability, 3),
        "lower_confidence_bound": round(lower, 3),
        "risk_level": risk,
        "expected_result": (
            f"{expected} {constraint} {visual_warning} {uncertainty_reason}"
        ),
        "observations": total,
        "feasible": feasible,
        "visual_evidence_status": evidence_status,
        "visual_uncertainty_count": len(uncertainties),
        "uncertainty_reason": uncertainty_reason,
    }