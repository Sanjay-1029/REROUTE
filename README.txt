REROUTE M4 visual provenance patch

Replace these three files in the REROUTE project root:
- meta_reasoning_adapter.py
- decision_pipeline.py
- visual_evidence.py

Back up the originals first. Do not replace dashboard.py, simulation_controller.py,
agent.py, or recovery files.

Then run:
python -m py_compile meta_reasoning_adapter.py decision_pipeline.py visual_evidence.py
python -m unittest discover -v

No live API request is made by applying or testing this patch.

Expected behaviour:
- The adapter marks image input attached only after an actual image data URL exists.
- Visual analysis is marked complete only when an image was attached and a non-empty
  visual_summary was returned.
- Offline mock summaries do not count as image analysis.
- Existing IMAGE_ANALYZED context remains supported.
- Navigation coordinates remain unverified.
