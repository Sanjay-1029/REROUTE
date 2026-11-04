# REROUTE final integration validation

## What is integrated

1. **Muse Spark API + image input:** the adapter sends an optional local image and structured navigation context in one Responses API request. It validates the returned JSON schema.
2. **Visual observations + navigation:** the agent passes its current position, goal, obstacle map, failure diagnosis, route availability, and optional image path into the reasoning policy. Image-relative observations are explicitly not treated as grid coordinates.
3. **Uncertainty:** Muse Spark's confidence, visual summary, and uncertainty list are retained. Historical action estimates include sample counts and a conservative Wilson lower confidence bound.
4. **Outcome prediction:** each recovery action is evaluated against current feasibility constraints and offline historical outcomes. An action with no evidence is marked `UNKNOWN`, not presented as proven.
5. **Decision validation:** model recommendations must pass schema and environment checks. The selector never lets a model preference override an infeasible, `HIGH`, or `UNKNOWN`-risk prediction.
6. **Predictive decision validation:** offline tests cover unsafe replanning, blocked retries, unknown/high-risk rejection, and pipeline action selection.
7. **Autonomous execution and verification:** the existing agent executes bounded recovery actions, replans from the live simulation state, verifies movement, records recovery outcomes, and stops safely when it cannot establish a safe action.

## Validation performed in the build environment

- `python -m compileall -q .` — passed.
- `python -m unittest discover -v` — 9 tests passed.
- `python main.py` — all three dynamic scenarios reached the goal; temporary obstacle expiry was observed.
- `python milestone11_evaluation.py` — 450 trials completed; initial environments were equivalent and two full runs were byte-equivalent. The candidate milestone 11 strategy recorded a 66.7% task-completion rate and 66.7% recovery success rate in this deterministic test set. These are simulation results, not real-world performance claims.
- Multimodal adapter tests mock HTTP. They verify image encoding and text-only payload construction without spending API credits.

## Important scope boundary

This validates the **software-in-the-loop grid simulation**, not a physical robot or a calibrated camera-to-grid system. A single image can describe visible hazards, but REROUTE does not yet have a camera calibration, depth sensor, localization stack, or robot motor driver that maps pixels to real-world coordinates and safely actuates hardware. Do not claim physical-world autonomy until those components are integrated and tested on hardware.

The successful one-request Muse Spark image test was run separately by the project owner. The automated test suite does not make live API calls. `python main.py --live` is capped at one live API request per run; remaining recovery decisions use the offline provider.

## Run

```powershell
python -m unittest discover -v
python main.py
```

Optional live smoke run (one API request maximum; requires `MODEL_API_KEY` in `.env`):

```powershell
python main.py --live --image test_image.jpg
```

Never commit `.env`, API keys, virtual environments, or private images to a public repository.
