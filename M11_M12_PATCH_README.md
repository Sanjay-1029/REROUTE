# REROUTE M11 + M12 Integrated Patch

## Included
- M11: persistent local storage for verified adaptive outcomes.
- M12: comparative candidate-action scoring using success estimates, failure exposure, execution cost, and unresolved uncertainty.
- Updated decision pipeline and selector integration.
- Regression tests for persistence and predictive scoring.

## Install
1. Back up your current REROUTE project folder.
2. Extract this ZIP into the project root, preserving folder structure, and overwrite the included files.
3. Run the validation command below from the project root.

```powershell
python -m compileall -q .
if ($LASTEXITCODE -eq 0) {
    python -m unittest discover -v
}
```

Expected result: 36 tests pass (34 existing + 2 added).

## M11 storage
By default, the simulation controller stores verified learning locally at:
`%USERPROFILE%\.reroute\adaptive_memory.json`

Only verified terminal outcomes are stored. No images, prompts, API keys, or personal data are saved.
Set `REROUTE_MEMORY_PATH` to choose another path.
Set `REROUTE_DISABLE_PERSISTENCE=1` to disable persistent storage.

Standalone `AdaptiveOutcomeMemory()` remains in-memory by default so tests and isolated experiments do not share history. The dashboard's SimulationController explicitly enables persistence.

## M12 limitations
M12 adds transparent comparative utility scores to candidate predictions. These are heuristic scores, not calibrated probabilities or guarantees about real-world outcomes. Existing feasibility checks and HIGH/UNKNOWN risk rejection remain in force. Model recommendations cannot bypass those gates.

## Validation performed on the supplied source package
- Python compileall: passed.
- Offline unittest suite: 36/36 passed.
- Existing 450-trial M11 evaluation: ran successfully and remained byte-equivalent reproducible.
- No live Meta API requests were made.
