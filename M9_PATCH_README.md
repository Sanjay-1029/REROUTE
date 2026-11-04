# REROUTE M9 Closed-Loop Re-observation Patch

## Files to replace
- `agent.py`
- `outcome_verifier.py`
- `test_agent_decision_integration.py`
- `test_m4_live.py`

## M9 behavior added
When the external decision pipeline selects `GATHER_INFORMATION`, the agent now:
1. Captures the current simulation snapshot.
2. Calls `image_provider` again for a post-action observation.
3. Runs the returned frame through the configured local vision engine.
4. Stores an auditable visual-observation history entry (`PRE_DECISION`, `POST_ACTION_REOBSERVATION`).
5. Emits an `OBSERVING` event for dashboard/event consumers.
6. Adds `OBSERVATION_REFRESHED` to the recovery verification record only when a post-action image was actually acquired.
7. Keeps acquisition failures explicit and does not claim that visual evidence was refreshed.

The existing verified movement, route replanning, progress-restoration, and task-completion checks remain intact. Pixel positions are not converted into map coordinates.

## Additional reliability fix
`test_m4_live.py` previously performed a live Muse Spark API request during module import. This made ordinary `unittest discover` unexpectedly attempt a live request. Its request and output are now inside `main()` and run only when the file is executed directly:
`python test_m4_live.py`

Importing/discovering the file no longer makes an API request.

## Validation performed against the supplied M9 source ZIP
- `python -m compileall -q .` — passed.
- `python -m unittest test_agent_decision_integration -v` — 5 tests passed.
- `python -m unittest discover -v` — 27 tests passed.
- No live Muse Spark API calls were made during validation.

## Apply
Extract these files into the project root and overwrite the matching files. Keep `.env` untouched. Keep dashboard live reasoning disabled for offline validation.
