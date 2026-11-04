import json
from pathlib import Path
from meta_reasoning_adapter import get_recovery_recommendation
from visual_evidence import assess_visual_evidence


def main():
    image_path = Path("test_image.jpg").resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    context = {
        "task": "Identify environmental obstacles and assess navigation safety.",
        "agent_status": "STATIONARY",
        "current_position": "Unknown",
        "visual_observations": [],
        "perception_status": "IMAGE_READY_FOR_MUSE_SPARK",
        "known_constraints": [
            "Do not invent navigation coordinates.",
            "Do not execute movement.",
            "Distinguish visible evidence from assumptions."
        ]
    }

    print("\nREROUTE M4: LIVE VISUAL REASONING TEST")
    print("Image:", image_path.name)
    print("Sending one request to Muse Spark...\n")

    recommendation = get_recovery_recommendation(
        context,
        image_path=image_path
    )

    print("MUSE SPARK RECOMMENDATION")
    print(json.dumps(recommendation, indent=2))

    context["perception_status"] = "IMAGE_ANALYZED"
    context["visual_observations"] = [
        recommendation.get("visual_summary", "")
    ]

    evidence = assess_visual_evidence(context, recommendation)

    print("\nVISUAL EVIDENCE ASSESSMENT")
    print(json.dumps(evidence, indent=2))

    print("\nSAFETY CHECK")
    print("Recommendation received:", True)
    print("Action executed:", False)
    print("Agent movement:", False)
    print("Navigation coordinates verified:",
          evidence["navigation_coordinates_verified"])

    print("\nM4 CONTROLLED TEST COMPLETED")


if __name__ == "__main__":
    main()
