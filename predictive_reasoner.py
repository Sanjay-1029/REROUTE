"""M12: transparent counterfactual scoring for candidate recovery actions.

This layer ranks only actions already evaluated by the evidence-based predictor.
It does not create new capabilities, override feasibility, or weaken risk gates.
Scores are comparative heuristics, not calibrated real-world probabilities.
"""
from __future__ import annotations

ACTION_COST = {
    "REPLAN": 8.0,
    "WAIT": 5.0,
    "RETRY": 10.0,
    "GATHER_INFORMATION": 6.0,
    "REQUEST_HUMAN_HELP": 20.0,
}


def assess_candidate(action: str, prediction: dict, environment_state: dict) -> dict:
    probability = float(prediction.get("success_probability", 0.0))
    feasible = prediction.get("feasible", False) is not False
    uncertainty_count = int(prediction.get("visual_uncertainty_count", 0) or 0)
    evidence_status = prediction.get("visual_evidence_status", "NO_VERIFIED_VISUAL_INPUT")

    # A conservative comparative utility: verified success value minus failure
    # exposure, execution cost, and unresolved uncertainty.
    success_value = 100.0 * probability
    failure_exposure = 45.0 * (1.0 - probability)
    execution_cost = ACTION_COST.get(action, 25.0)
    uncertainty_penalty = min(uncertainty_count, 5) * 3.0
    if evidence_status != "VISUAL_EVIDENCE_ANALYZED":
        uncertainty_penalty += 8.0

    information_gain = 0.0
    if action == "GATHER_INFORMATION":
        # Potential information value only; no claim that the sensor will succeed.
        information_gain = min(max(uncertainty_count, 1), 3) * 5.0

    utility = success_value - failure_exposure - execution_cost - uncertainty_penalty + information_gain
    if not feasible:
        utility = -1000.0

    return {
        "expected_utility": round(utility, 2),
        "success_value": round(success_value, 2),
        "failure_exposure": round(failure_exposure, 2),
        "execution_cost": execution_cost,
        "uncertainty_penalty": round(uncertainty_penalty, 2),
        "potential_information_value": round(information_gain, 2),
        "assumption": "Comparative heuristic; not a calibrated real-world outcome probability.",
    }


def rank_candidate_actions(evaluated_actions: list[dict], environment_state: dict) -> list[dict]:
    ranked = []
    for item in evaluated_actions:
        enriched = dict(item)
        prediction = dict(item.get("prediction", {}))
        prediction["predictive_assessment"] = assess_candidate(
            item.get("action", ""), prediction, environment_state or {}
        )
        enriched["prediction"] = prediction
        ranked.append(enriched)
    ranked.sort(
        key=lambda item: (
            item["prediction"]["predictive_assessment"]["expected_utility"],
            item["action"],
        ),
        reverse=True,
    )
    for index, item in enumerate(ranked, start=1):
        item["prediction"]["predictive_assessment"]["rank"] = index
    return ranked
