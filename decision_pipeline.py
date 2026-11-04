"""Evidence -> recommendation -> validation -> prediction -> safe selection."""
from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable

from meta_reasoning_adapter import get_recovery_recommendation
from recommendation_validator import validate_recommendation
from outcome_predictor import predict_outcome, build_action_statistics
from decision_selector import select_best_action
from evaluation import EvaluationRunner
from visual_evidence import assess_visual_evidence
from predictive_reasoner import rank_candidate_actions


@lru_cache(maxsize=1)
def _historical_action_statistics():
    """Run deterministic offline evaluation once per process."""
    return build_action_statistics(EvaluationRunner().run())


def _build_evidence_context(context: dict[str, Any], recommendation: dict[str, Any]) -> dict[str, Any]:
    """Merge adapter-verified visual provenance without inventing map facts."""
    evidence_context = dict(context)
    attached = recommendation.get("_reroute_visual_input_attached") is True
    analyzed = recommendation.get("_reroute_visual_analysis_completed") is True and attached
    summary = recommendation.get("visual_summary")

    if analyzed:
        observations = evidence_context.get("visual_observations", [])
        if not isinstance(observations, list):
            observations = []
        observations = list(observations)
        if isinstance(summary, str) and summary.strip() and summary not in observations:
            observations.append(summary)
        evidence_context["visual_observations"] = observations
        evidence_context["perception_status"] = "IMAGE_ANALYZED"

    return evidence_context


def run_decision_pipeline(
    context: dict[str, Any],
    recommendation_provider: Callable | None = None,
) -> dict[str, Any]:
    if not isinstance(context, dict):
        raise TypeError("context must be a dictionary.")

    statistics = {
        action: dict(counts)
        for action, counts in _historical_action_statistics().items()
    }
    # Verified runtime outcomes supplement the deterministic baseline. They
    # never bypass feasibility checks or the selector's risk gates.
    adaptive = context.get("adaptive_learning", {})
    adaptive_stats = adaptive.get("action_statistics", {}) if isinstance(adaptive, dict) else {}
    if isinstance(adaptive_stats, dict):
        for action, counts in adaptive_stats.items():
            if not isinstance(counts, dict):
                continue
            current = statistics.setdefault(action, {"success": 0, "failure": 0})
            for outcome in ("success", "failure"):
                value = counts.get(outcome, 0)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    current[outcome] = current.get(outcome, 0) + value

    provider = recommendation_provider or get_recovery_recommendation
    recommendation = provider(context)
    valid, message = validate_recommendation(recommendation, context)

    if not valid:
        return {"status": "REJECTED", "reason": message, "original_recommendation": recommendation}

    evidence_context = _build_evidence_context(context, recommendation)
    visual_evidence = assess_visual_evidence(evidence_context, recommendation)
    environment_state = {
        **context,
        **context.get("environment_state", {}),
        "visual_evidence": visual_evidence,
        "adaptive_learning": {
            "source": adaptive.get("source", "NO_RUNTIME_HISTORY") if isinstance(adaptive, dict) else "NO_RUNTIME_HISTORY",
            "diagnosis": adaptive.get("diagnosis", "ALL") if isinstance(adaptive, dict) else "ALL",
            "verified_observations": adaptive.get("observations", 0) if isinstance(adaptive, dict) else 0,
            "total_verified_outcomes": adaptive.get("total_verified_outcomes", 0) if isinstance(adaptive, dict) else 0,
        },
    }

    evaluated = []
    for action in ("REPLAN", "WAIT", "RETRY", "GATHER_INFORMATION", "REQUEST_HUMAN_HELP"):
        prediction = predict_outcome(action, environment_state, statistics)
        evaluated.append({"action": action, "prediction": prediction})

    # M12 compares counterfactual action outcomes. The selector still applies
    # its existing feasibility and risk gates before any candidate can win.
    evaluated = rank_candidate_actions(evaluated, environment_state)
    selected = select_best_action(
        evaluated,
        recommended_action=recommendation["recommended_action"],
    )

    if selected is None:
        return {
            "status": "NO_SAFE_ACTION",
            "reason": (
                "No candidate action has sufficient historical support "
                "and passes current safety constraints."
            ),
            "original_recommendation": recommendation,
            "evaluated_actions": evaluated,
            "visual_evidence": visual_evidence,
        }

    return {
        "status": "EVALUATED",
        "original_recommendation": recommendation,
        "selected_action": selected["action"],
        "prediction": selected["prediction"],
        "evaluated_actions": evaluated,
        "visual_summary": recommendation.get("visual_summary", ""),
        "uncertainties": recommendation.get("uncertainties", []),
        "required_checks": recommendation.get("required_checks", []),
        "visual_evidence": visual_evidence,
        "adaptive_learning": {
            "source": adaptive.get("source", "NO_RUNTIME_HISTORY") if isinstance(adaptive, dict) else "NO_RUNTIME_HISTORY",
            "diagnosis": adaptive.get("diagnosis", "ALL") if isinstance(adaptive, dict) else "ALL",
            "verified_observations": adaptive.get("observations", 0) if isinstance(adaptive, dict) else 0,
            "total_verified_outcomes": adaptive.get("total_verified_outcomes", 0) if isinstance(adaptive, dict) else 0,
        },
    }
