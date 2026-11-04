"""Offline tests for Muse Spark image request construction; no real HTTP calls."""
import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import meta_reasoning_adapter as adapter

class FakeResponse:
    def raise_for_status(self): pass
    def json(self):
        recommendation = {
            "recommended_action": "REPLAN", "reason": "Visible obstruction; alternate route available.",
            "confidence": 0.88, "required_checks": ["Verify the alternate route."],
            "alternative_action": "GATHER_INFORMATION", "visual_summary": "A chair is visible.",
            "uncertainties": ["Image coordinates are not map coordinates."],
        }
        return {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(recommendation)}]}]}

class MultimodalAdapterTests(unittest.TestCase):
    def test_local_image_is_encoded_as_responses_input_image(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scene.jpg"
            raw = b"offline-test-image-bytes"
            path.write_bytes(raw)
            with patch.dict(os.environ, {"MODEL_API_KEY": "not-a-real-key"}):
                with patch.object(adapter.requests, "post", return_value=FakeResponse()) as post:
                    result = adapter.get_recovery_recommendation({"failure_type": "MOVEMENT_BLOCKED"}, image_path=path)
            self.assertEqual(result["recommended_action"], "REPLAN")
            self.assertEqual(post.call_count, 1)
            payload = post.call_args.kwargs["json"]
            self.assertEqual(payload["model"], "muse-spark-1.3")
            content = payload["input"][0]["content"]
            image_block = next(block for block in content if block["type"] == "input_image")
            self.assertEqual(base64.b64decode(image_block["image_url"].split(",", 1)[1]), raw)

    def test_text_only_request_when_no_image_configured(self):
        with patch.dict(os.environ, {"MODEL_API_KEY": "not-a-real-key", "REROUTE_IMAGE_PATH": ""}):
            with patch.object(adapter.requests, "post", return_value=FakeResponse()) as post:
                adapter.get_recovery_recommendation({"failure_type": "UNKNOWN"})
        blocks = post.call_args.kwargs["json"]["input"][0]["content"]
        self.assertEqual([block["type"] for block in blocks], ["input_text"])

if __name__ == "__main__":
    unittest.main(verbosity=2)
