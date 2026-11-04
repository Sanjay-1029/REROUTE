# REROUTE

**REROUTE** is an uncertainty-aware autonomous navigation and recovery agent designed to help intelligent systems adapt when unexpected situations occur.

It combines environment-state analysis, visual evidence, AI-assisted reasoning, action validation, predictive evaluation, recovery planning and outcome verification.

## Key Capabilities

- Detects and diagnoses navigation failures.
- Evaluates multiple possible recovery actions.
- Integrates Meta Muse Spark through the Meta Model API.
- Supports visual evidence analysis.
- Independently evaluates AI recommendations before selecting actions.
- Uses historical outcomes to support predictive decision-making.
- Verifies execution outcomes and goal completion.

## Technology Stack

- Python
- PyBullet
- A* pathfinding
- Meta Muse Spark
- Meta Model API

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a `.env` file for live Muse Spark integration:

```text
MODEL_API_KEY=your_key_here
```

Never commit or share your actual API key.

## Run Offline

```powershell
python -m unittest discover -v
python main.py
python milestone11_evaluation.py
```

Offline mode uses a mock reasoning provider and does not make Meta API calls.

## Live Muse Spark Test

```powershell
python live_muse_test.py --image test_image.jpg
```

This sends one image-and-context request to Muse Spark. It does not execute the recommended action.

## PyBullet Live Demonstration

```powershell
python .\run_pybullet_live_demo.py --live --image .\test_image.jpg
```

This demonstration integrates Muse Spark reasoning with the PyBullet navigation simulation.

The supplied image is static visual evidence, not a live camera feed from the simulator. Muse Spark provides a recommendation, which REROUTE independently evaluates before selecting a recovery action.

The live integration is configured to make a maximum of one live API request per run. Subsequent decisions may use the offline provider.

Live API usage may incur charges.

## Validation and Limitations

See `FINAL_VALIDATION.md` and `DEVELOPMENT_STATUS.md` for validation details.

The demonstrated environment is a simulated grid-world robotics environment. REROUTE is not yet deployed on a physical robot. Real-world camera calibration, localization, depth sensing and motor-control integration remain separate development milestones.