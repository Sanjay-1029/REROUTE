# REROUTE direct Muse Spark multimodal integration

## What changed

- `meta_reasoning_adapter.py` sends a local image directly to Muse Spark using a Responses API `input_image` block with a Base64 data URL.
- Text-only reasoning remains supported when no image is configured.
- `decision_pipeline.py` validates the model recommendation, evaluates candidate actions using existing historical simulation data, and returns the selected action to the autonomous agent.
- `simulation_controller.py` defaults to an offline mock provider. Live reasoning requires explicit opt-in with `SimulationController(live_reasoning=True)`.
- `main.py` defaults to offline mode. It does not call Meta unless `--live` is supplied.

## Setup

Keep the existing `.env` private and ensure it contains `MODEL_API_KEY=...`. Never paste the key into chat or commit `.env` to Git.

Install adapter dependencies if needed:

```powershell
pip install requests python-dotenv
```

## Offline tests (no Meta request)

```powershell
python -m unittest test_multimodal_adapter.py -v
python main.py
```

## Explicit live Muse Spark run

```powershell
python main.py --live --image "C:\path\to\scene.jpg"
```

Each recovery event may trigger a separate API request. Without `--image`, Muse Spark receives structured text context only. Supported inline image formats: JPEG, PNG, WEBP, GIF; maximum inline image size: 50 MB.

## Safety and limitations

- This integration does not call YOLO. Muse Spark receives the image directly.
- A single image does not automatically provide metric depth or calibrated world coordinates. Do not treat image-relative locations as navigation-grid coordinates without calibration or a map transform.
- Recommendations are validated and evaluated before being returned to the agent. The agent still executes supported recovery actions and verifies movement.
- The automated tests mock HTTP responses; they do not validate live API credentials, availability, or billing.


## Live Multimodal Validation

**Status:** PASSED

**Model:** Muse Spark 1.3

**Test image:** test_image.jpg

**API request:** One successful live request.

**Observed capabilities:**

* Successfully transmitted a local image to Muse Spark.
* Identified a wet-floor warning sign and reflective flooring.
* Connected visual evidence with structured navigation context.
* Recommended REPLAN when the next waypoint was reported blocked.
* Returned confidence, required checks, alternative action, visual summary, and uncertainties.

**Model recommendation:** REPLAN

**Confidence:** 0.82

**Important limitation:** Camera-to-grid coordinate calibration has not yet been implemented or validated.

**Execution:** The recommendation was not executed. This was a reasoning-only test.
