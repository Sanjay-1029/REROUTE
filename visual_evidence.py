"""Structured assessment of visual evidence from Muse Spark."""


def assess_visual_evidence(context, recommendation):
    """Assess visual evidence without treating image descriptions as map facts."""
    if not isinstance(context, dict):
        raise TypeError("Context must be a dictionary.")
    if not isinstance(recommendation, dict):
        raise TypeError("Recommendation must be a dictionary.")

    observations = context.get("visual_observations", [])
    uncertainties = recommendation.get("uncertainties", [])
    perception_status = context.get("perception_status", "UNKNOWN")

    if not isinstance(observations, list):
        observations = []
    if not isinstance(uncertainties, list):
        uncertainties = []

    image_available = perception_status in {"IMAGE_READY_FOR_MUSE_SPARK", "IMAGE_ANALYZED"}
    analysis_completed = perception_status == "IMAGE_ANALYZED"

    return {
        "image_available": image_available,
        "analysis_completed": analysis_completed,
        "observation_count": len(observations),
        "observations": observations,
        "uncertainties": uncertainties,
        "evidence_status": (
            "VISUAL_EVIDENCE_ANALYZED"
            if analysis_completed
            else "IMAGE_AVAILABLE_NOT_ANALYZED"
            if image_available
            else "NO_VERIFIED_VISUAL_INPUT"
        ),
        "navigation_coordinates_verified": False,
    }
