from meta_reasoning_adapter import get_recovery_recommendation
from recommendation_validator import validate_recommendation


def main():

    context = {
        "failure_type": "MOVEMENT_BLOCKED",
        "failure_evidence": [
            "Forward movement blocked by a dynamic obstacle."
        ],
        "diagnosis": "Current route is blocked.",
        "confidence": 0.90,
        "environment_state": {
            "agent_position": [2, 3],
            "goal_position": [8, 8],
            "obstacle_position": [3, 3]
        },
        "safe_alternative_route": True,
        "obstacles_known": True,
        "path_blocked": True,
        "recovery_history": []
    }

    print("1. Sending situation to Meta AI...")

    try:
        recommendation = get_recovery_recommendation(context)

        print("\n2. Meta AI Recommendation:")
        print(recommendation)

        print("\n3. Validating recommendation...")

        is_valid, message = validate_recommendation(
            recommendation,
            context
        )

        print("\n4. Validation Result:")
        print("Valid:", is_valid)
        print("Message:", message)

        if is_valid:
            print("\nREROUTE: Recommendation accepted for further evaluation.")
        else:
            print("\nREROUTE: Recommendation rejected.")

    except Exception as error:
        print("\nPipeline Error:", error)


if __name__ == "__main__":
    main()