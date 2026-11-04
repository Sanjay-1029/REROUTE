"""Runtime integration tests for the external decision pipeline."""
import unittest

from agent import AutonomousAgent
from environment import Environment
from failure_detector import FailureDetector


class DecisionExecutionTests(unittest.TestCase):

    def create_agent_with_failure(self, selected_action):
        env = Environment(
            width=5,
            height=5,
            agent_position=(0, 0),
            goal_position=(4, 4),
            obstacles=[(1, 0)]
        )

        calls = []

        def external_policy(context):
            calls.append(context)
            return {
                "selected_action": selected_action
            }

        agent = AutonomousAgent(
            env,
            logger=lambda _: None,
            recovery_mode="m11",
            decision_policy=external_policy
        )

        movement = env.move("right")

        detector = FailureDetector()
        failure = detector.observe_movement(
            env,
            movement,
            (1, 0)
        )

        return agent, failure, calls

    def test_external_gather_information_executes(self):
        agent, failure, calls = self.create_agent_with_failure(
            "GATHER_INFORMATION"
        )

        result = agent._recover(failure, (1, 0))

        self.assertTrue(result)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(agent.recovery_engine.history), 1)

        record = agent.recovery_engine.history[0]

        self.assertEqual(
            record.decision.action.value,
            "GATHER_INFORMATION"
        )
        self.assertTrue(record.succeeded)
        self.assertIn("snapshot_v", record.execution_result)

    def test_external_replan_executes(self):
        agent, failure, calls = self.create_agent_with_failure(
            "REPLAN"
        )

        result = agent._recover(failure, (1, 0))

        self.assertTrue(result)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(agent.recovery_engine.history), 1)

        record = agent.recovery_engine.history[0]

        self.assertEqual(record.decision.action.value, "REPLAN")
        self.assertTrue(record.succeeded)
        self.assertEqual(
            record.execution_result,
            "alternative_route_available"
        )


    def test_fresh_image_provider_runs_local_vision_and_reaches_policy(self):
        import os
        import tempfile

        agent, failure, calls = self.create_agent_with_failure("GATHER_INFORMATION")

        class FakeVisionEngine:
            def load_image(self, image_path):
                return object()

            def prepare_reasoning_context(self, navigation_context, image):
                return {
                    **navigation_context,
                    "visual_observations": [{
                        "object": "chair",
                        "position": "CENTER",
                        "obstacle_candidate": True,
                        "requires_verification": True,
                    }],
                    "visual_detection_count": 1,
                    "perception_status": "OBJECTS_DETECTED",
                    "perception_source": "YOLO",
                    "perception_limitation": "Image-relative positions are not grid coordinates.",
                }

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as image_file:
            image_path = image_file.name

        try:
            provider_calls = []
            agent.vision_engine = FakeVisionEngine()
            agent.image_provider = lambda: provider_calls.append(True) or image_path

            self.assertTrue(agent._recover(failure, (1, 0)))
            self.assertEqual(len(provider_calls), 2)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["perception_status"], "IMAGE_READY_FOR_MUSE_SPARK")
            self.assertEqual(calls[0]["visual_detection_count"], 1)
            self.assertEqual(calls[0]["visual_observations"][0]["object"], "chair")
            self.assertEqual(calls[0]["image_path"], image_path)
            self.assertIn("not grid coordinates", calls[0]["perception_limitation"])
            self.assertEqual(len(agent.visual_observation_history), 2)
            self.assertEqual(
                agent.visual_observation_history[0]["phase"],
                "PRE_DECISION",
            )
            self.assertEqual(
                agent.visual_observation_history[1]["phase"],
                "POST_ACTION_REOBSERVATION",
            )
            record = agent.recovery_engine.history[0]
            self.assertIn(
                "OBSERVATION_REFRESHED",
                [state.value for state in record.verification_states],
            )
        finally:
            os.unlink(image_path)

    def test_post_action_reobservation_failure_is_recorded_without_claiming_refresh(self):
        import os
        import tempfile

        agent, failure, calls = self.create_agent_with_failure("GATHER_INFORMATION")

        class FakeVisionEngine:
            def load_image(self, image_path):
                return object()

            def prepare_reasoning_context(self, navigation_context, image):
                return {
                    **navigation_context,
                    "visual_observations": [{"object": "chair", "position": "CENTER"}],
                    "visual_detection_count": 1,
                    "perception_status": "OBJECTS_DETECTED",
                    "perception_source": "YOLO",
                }

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as image_file:
            image_path = image_file.name

        provider_calls = []

        def provider():
            provider_calls.append(True)
            if len(provider_calls) == 1:
                return image_path
            raise RuntimeError("camera disconnected after action")

        try:
            agent.vision_engine = FakeVisionEngine()
            agent.image_provider = provider
            self.assertTrue(agent._recover(failure, (1, 0)))
            self.assertEqual(len(provider_calls), 2)
            self.assertEqual(calls[0]["perception_status"], "IMAGE_READY_FOR_MUSE_SPARK")
            self.assertEqual(
                agent.last_visual_context["perception_status"],
                "IMAGE_ACQUISITION_FAILED",
            )
            self.assertEqual(
                agent.visual_observation_history[-1]["perception_status"],
                "IMAGE_ACQUISITION_FAILED",
            )
            record = agent.recovery_engine.history[0]
            self.assertNotIn(
                "OBSERVATION_REFRESHED",
                [state.value for state in record.verification_states],
            )
        finally:
            os.unlink(image_path)

    def test_image_provider_failure_does_not_claim_visual_evidence(self):
        agent, failure, calls = self.create_agent_with_failure("GATHER_INFORMATION")
        agent.image_provider = lambda: (_ for _ in ()).throw(RuntimeError("camera unavailable"))

        self.assertTrue(agent._recover(failure, (1, 0)))
        self.assertEqual(calls[0]["perception_status"], "IMAGE_ACQUISITION_FAILED")
        self.assertEqual(calls[0]["visual_observations"], [])
        self.assertIsNone(calls[0]["image_path"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
