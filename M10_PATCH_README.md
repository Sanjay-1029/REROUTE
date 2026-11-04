# REROUTE M10 — Verified Adaptive Outcome Learning

## Files to replace/add

Replace:
- `agent.py`
- `decision_pipeline.py`
- `simulation_controller.py`

Add:
- `adaptive_learning.py`
- `test_adaptive_learning.py`

## What M10 adds

1. A lightweight adaptive outcome memory shared by simulation runs handled by the same `SimulationController` instance.
2. Diagnosis-specific action statistics built only from terminal, verified recovery outcomes.
3. Feedback rules:
   - `WAIT` success requires verified environment change.
   - `GATHER_INFORMATION` success requires an actual post-action observation refresh.
   - `RETRY` and `REPLAN` success requires verified progress restoration or task completion.
   - Explicit recovery failures count as failures.
   - Unverified attempts are excluded.
   - `REQUEST_HUMAN_HELP` is excluded from autonomous action-performance statistics.
4. Adaptive counts supplement the existing deterministic historical baseline in `decision_pipeline.py`.
5. Existing feasibility checks, recommendation validation, uncertainty handling, and high/unknown-risk rejection remain in force.
6. Adaptive outcome totals are included in evaluated decision metadata.

## Scope and persistence

- Memory is in RAM and persists across runs using the same `SimulationController` instance.
- Restarting the application clears the adaptive memory. No file or database is written.
- This is verified outcome-based adaptation, not model fine-tuning or online training of Muse Spark.
- The offline grid simulation is the validation target. It does not establish physical robot performance.

## Validation performed against the supplied M10 source package

- `python -m compileall -q .` — passed.
- `python -m unittest discover -v` — 34 tests passed.
- `python milestone11_evaluation.py` — 450 trials, equivalent initial environments, byte-equivalent repeated results.
- Existing M11 candidate benchmark reproduced: 66.7% task completion, 66.7% recovery success, zero false-positive recoveries.
- No live Muse Spark API calls were made.

## Apply

Back up the current project first. Extract this ZIP into the project root and overwrite the three listed files; add the two new files. Do not replace `.env`, model weights, or other files.

Then run:

```powershell
python -m compileall -q .
python -m unittest discover -v
python milestone11_evaluation.py
```

Keep live reasoning disabled during validation.
