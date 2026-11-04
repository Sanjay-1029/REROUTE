"""Strict schema and environment-safety checks for model recommendations."""
import math

ALLOWED_ACTIONS = {"WAIT", "RETRY", "GATHER_INFORMATION", "REPLAN", "REQUEST_HUMAN_HELP"}


def validate_recommendation(recommendation, environment_state):
    if not isinstance(recommendation, dict):
        return False, "Invalid recommendation format."
    action = recommendation.get("recommended_action")
    if action not in ALLOWED_ACTIONS:
        return False, "Unsupported recovery action."
    confidence = recommendation.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        return False, "Invalid confidence value."
    if not isinstance(recommendation.get("reason"), str) or not recommendation["reason"].strip():
        return False, "Recommendation reason must be non-empty text."
    checks = recommendation.get("required_checks")
    if not isinstance(checks, list) or not checks or not all(isinstance(item, str) and item.strip() for item in checks):
        return False, "Required checks must be a non-empty list of strings."
    if recommendation.get("alternative_action") not in ALLOWED_ACTIONS:
        return False, "Unsupported alternative action."
    if not isinstance(environment_state, dict):
        return False, "Invalid environment state."
    if action == "REPLAN":
        if not environment_state.get("safe_alternative_route", False):
            return False, "No confirmed safe alternative route."
        if not environment_state.get("obstacles_known", False):
            return False, "Obstacle information is incomplete."
    if action == "RETRY" and environment_state.get("path_blocked", False):
        return False, "Cannot retry while the path remains blocked."
    if action == "WAIT" and environment_state.get("wait_budget_exhausted", False):
        return False, "Wait budget is exhausted."
    return True, "Environment-aware validation passed."
