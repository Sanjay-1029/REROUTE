"""Evidence-based selector with conservative safety constraints."""

ALLOWED_ACTIONS = {
    "WAIT",
    "RETRY",
    "GATHER_INFORMATION",
    "REPLAN",
    "REQUEST_HUMAN_HELP",
}

RISK_SCORES = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "UNKNOWN": 4,
}


def select_best_action(evaluated_actions, recommended_action=None):
    if not evaluated_actions:
        return None

    eligible = []

    for item in evaluated_actions:
        action = item.get("action")
        prediction = item.get("prediction", {})

        probability = prediction.get("success_probability")
        confidence = prediction.get("lower_confidence_bound", 0.0)
        risk = prediction.get("risk_level")

        if action not in ALLOWED_ACTIONS:
            continue

        if (
            not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or not 0 <= probability <= 1
        ):
            continue

        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 <= confidence <= 1
        ):
            continue

        if risk not in RISK_SCORES:
            continue

        if prediction.get("feasible", True) is False:
            continue

        if risk in {"HIGH", "UNKNOWN"}:
            continue

        eligible.append(item)

    if not eligible:
        return None

    eligible.sort(
        key=lambda item: (
            RISK_SCORES[item["prediction"]["risk_level"]],
            -item["prediction"].get("lower_confidence_bound", 0.0),
            -item["prediction"]["success_probability"],
            item["action"] != recommended_action,
            -item["prediction"].get("predictive_assessment", {}).get("expected_utility", 0.0),
            item["action"],
        )
    )

    return eligible[0]
