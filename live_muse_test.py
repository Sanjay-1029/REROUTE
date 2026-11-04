"""Make exactly one direct multimodal Muse Spark request; does not run scenarios."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from meta_reasoning_adapter import get_recovery_recommendation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send one image plus navigation context to Muse Spark."
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to one local JPEG, PNG, WEBP, or GIF scene image."
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    context = {
        "test_type": "single_live_multimodal_validation",
        "agent_position": [0, 0],
        "goal_position": [9, 9],
        "grid_dimensions": [10, 10],
        "known_obstacles": [[0, 1]],
        "current_route": [[0, 0], [0, 1], [0, 2]],
        "failure_type": "MOVEMENT_BLOCKED",
        "situation": (
            "The next planned cell is reported blocked. Inspect the attached image, "
            "compare visible evidence with the structured state, and recommend a "
            "safe recovery action. Do not infer metric distances or grid coordinates "
            "from image-relative positions."
        ),
        "image_path": str(image_path),
    }

    print("LIVE TEST: sending exactly one request to Muse Spark.")
    print(f"Image: {image_path.name} ({image_path.stat().st_size:,} bytes)")
    result = get_recovery_recommendation(context)
    print("\nMUSE SPARK RESPONSE:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nCompleted. This script does not execute the recommended action.")


if __name__ == "__main__":
    main()
