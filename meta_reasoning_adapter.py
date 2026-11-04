"""Meta Muse Spark adapter for text-only and direct multimodal reasoning."""
from __future__ import annotations
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any
import requests
from dotenv import load_dotenv

load_dotenv()
API_URL = "https://api.meta.ai/v1/responses"
MODEL = "muse-spark-1.3"
MAX_INLINE_IMAGE_BYTES = 50_000_000
ALLOWED_ACTIONS = {"WAIT", "RETRY", "GATHER_INFORMATION", "REPLAN", "REQUEST_HUMAN_HELP"}


def _build_image_data_url(image_path: str | os.PathLike[str]) -> str:
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"REROUTE image not found: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError("REROUTE image is empty.")
    if size > MAX_INLINE_IMAGE_BYTES:
        raise ValueError(f"Image exceeds the {MAX_INLINE_IMAGE_BYTES:,}-byte inline limit.")
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        raise ValueError("Unsupported image format. Use JPEG, PNG, WEBP, or GIF.")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _extract_output_text(data: dict[str, Any]) -> str:
    chunks = []
    for item in data.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text" and block.get("text"):
                    chunks.append(block["text"])
    if not chunks:
        raise ValueError("Muse Spark returned no output text.")
    return "\n".join(chunks).strip()


def _parse_recommendation(output_text: str) -> dict[str, Any]:
    text = output_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Muse Spark returned invalid JSON.") from exc
    if not isinstance(result, dict):
        raise ValueError("Muse Spark recommendation must be a JSON object.")
    required = {"recommended_action", "reason", "confidence", "required_checks", "alternative_action"}
    if not required.issubset(result):
        raise ValueError(f"Muse Spark response is missing fields: {sorted(required - set(result))}")
    if result["recommended_action"] not in ALLOWED_ACTIONS:
        raise ValueError("Muse Spark returned an unsupported recovery action.")
    if result["alternative_action"] not in ALLOWED_ACTIONS:
        raise ValueError("Muse Spark returned an unsupported alternative action.")
    confidence = result["confidence"]
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
        raise ValueError("Muse Spark confidence must be between 0 and 1.")
    if not isinstance(result["reason"], str) or not result["reason"].strip():
        raise ValueError("Muse Spark reason must be non-empty text.")
    checks = result["required_checks"]
    if not isinstance(checks, list) or not all(isinstance(x, str) for x in checks):
        raise ValueError("Muse Spark required_checks must be a list of strings.")
    return result


def get_recovery_recommendation(
    context: dict[str, Any],
    image_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Request a recovery decision and record whether visual input was truly used."""
    if not isinstance(context, dict):
        raise TypeError("context must be a dictionary.")
    api_key = os.getenv("MODEL_API_KEY")
    if not api_key:
        raise RuntimeError("MODEL_API_KEY not found in environment or .env")

    selected_image = image_path or context.get("image_path") or os.getenv("REROUTE_IMAGE_PATH")
    prompt_context = {k: v for k, v in context.items() if k not in {"image_path", "image_data_url", "image_bytes"}}
    attached_intent = bool(selected_image or context.get("image_data_url"))
    prompt_context["visual_input"] = (
        "An image is attached. Inspect it directly. Distinguish visible evidence from assumptions. "
        "Do not convert image-relative locations into map/grid coordinates without a supplied calibration."
        if attached_intent else "No image supplied; reason only from structured environment evidence."
    )
    prompt = f"""You are the perception and reasoning component of REROUTE, an uncertainty-aware autonomous agent.
Analyze the structured environment evidence and, if attached, the image itself. Distinguish observations,
inferences, and unknowns. Never invent object coordinates, distances, or map positions. Image-relative
locations are not navigation-grid coordinates. If evidence is insufficient or conflicting, prefer
GATHER_INFORMATION or REQUEST_HUMAN_HELP over unsafe action.

Recommend exactly one action: WAIT, RETRY, GATHER_INFORMATION, REPLAN, REQUEST_HUMAN_HELP.

CONTEXT:
{json.dumps(prompt_context, indent=2, default=str)}

Return ONLY valid JSON with this structure:
{{"recommended_action":"REPLAN","reason":"Evidence-based explanation.","confidence":0.0,
"required_checks":["Concrete verification step."],"alternative_action":"GATHER_INFORMATION",
"visual_summary":"What the image supports, or No image supplied.","uncertainties":["Important unknowns."]}}
"""
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    data_url = context.get("image_data_url")
    if data_url:
        if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
            raise ValueError("image_data_url must be a valid image data URL.")
    elif selected_image:
        data_url = _build_image_data_url(selected_image)

    image_attached = bool(data_url)
    if image_attached:
        content.append({"type": "input_image", "image_url": data_url})

    payload = {"model": MODEL, "input": [{"type": "message", "role": "user", "content": content}]}
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=90,
    )
    response.raise_for_status()
    response_data = response.json()
    if response_data.get("status") not in {None, "completed"}:
        raise RuntimeError(f"Muse Spark request did not complete: {response_data.get('status')}")

    recommendation = _parse_recommendation(_extract_output_text(response_data))
    summary = recommendation.get("visual_summary")
    summary_present = isinstance(summary, str) and bool(summary.strip())

    # These flags are adapter-generated, not model-controlled claims.
    recommendation["_reroute_visual_input_attached"] = image_attached
    recommendation["_reroute_visual_analysis_completed"] = image_attached and summary_present
    return recommendation
