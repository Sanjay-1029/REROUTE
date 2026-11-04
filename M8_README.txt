REROUTE M8 — Autonomous Visual Observation and Evidence Acquisition
===================================================================

Files to replace in the project root:
- agent.py
- simulation_controller.py
- test_agent_decision_integration.py

No changes are made to vision_engine.py, decision_pipeline.py,
meta_reasoning_adapter.py, or visual_evidence.py. The patch uses their
existing interfaces and preserves the M4 visual-provenance checks.

What this adds
--------------
1. AutonomousAgent calls image_provider during recovery to acquire a fresh
   frame. The provider may return:
   - a filesystem path to an image, or
   - an OpenCV/numpy image array.
2. If a VisionEngine is injected, it runs local YOLO candidate detection and
   contributes observations and uncertainty context.
3. In-memory frames are JPEG-encoded as a data URL for the existing
   multimodal Muse Spark adapter. File paths are forwarded through the
   existing image_path interface.
4. Visual candidates are explicitly not treated as confirmed obstacles or
   converted to map/grid coordinates.
5. Acquisition/processing failures are represented as failure statuses and
   do not claim verified visual evidence.
6. SimulationController accepts optional vision_engine and image_provider
   dependencies and forwards them to AutonomousAgent.
7. Live reasoning remains OFF by default. The existing one-live-request-per-run
   guard is unchanged.

Example integration
-------------------
    from vision_engine import VisionEngine
    from simulation_controller import SimulationController

    vision = VisionEngine()
    controller = SimulationController(
        live_reasoning=False,  # Keep offline while validating M8
        vision_engine=vision,
        image_provider=my_camera_frame_provider,
    )

`my_camera_frame_provider` must return one fresh OpenCV/numpy frame per call.
Do not enable live_reasoning until offline validation is complete and a
controlled live request is explicitly authorized.

Validation
----------
From the project root with the virtual environment active:

    python -m py_compile agent.py simulation_controller.py test_agent_decision_integration.py
    python -m unittest test_agent_decision_integration -v
    python -m unittest discover -v

The patch was syntax-compiled in the preparation environment. The complete
project test suite must be run in the user's project environment because the
uploaded ZIP contained only the seven named source files, not the full project.
