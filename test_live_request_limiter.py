"""Offline tests for the single-live-request limiter."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import simulation_controller


class FakeAgent:
    def __init__(self, environment, **kwargs):
        self.decision_policy = kwargs["decision_policy"]

    def run(self):
        return SimpleNamespace(reached_goal=True, reason=None)


class LiveRequestLimiterTests(unittest.TestCase):

    def create_controller(self):
        controller = simulation_controller.SimulationController(
            live_reasoning=True
        )

        with patch.object(simulation_controller, "AutonomousAgent", FakeAgent):
            controller.start(scenario=next(iter(controller.factories)))

        if controller._thread:
            controller._thread.join(timeout=2)

        return controller, controller._agent.decision_policy

    @patch("simulation_controller.reroute_decision_policy")
    def test_only_first_attempt_uses_live_reasoning(self, mock_policy):
        mock_policy.return_value = {"selected_action": "WAIT"}

        controller, policy = self.create_controller()

        policy({})
        policy({})

        self.assertEqual(
            [call.kwargs["live_reasoning"] for call in mock_policy.call_args_list],
            [True, False]
        )
        self.assertTrue(controller._live_request_used)

    @patch("simulation_controller.reroute_decision_policy")
    def test_failed_live_attempt_still_consumes_allowance(self, mock_policy):
        mock_policy.side_effect = [
            RuntimeError("Simulated API failure"),
            {"selected_action": "WAIT"}
        ]

        controller, policy = self.create_controller()

        with self.assertRaises(RuntimeError):
            policy({})

        result = policy({})

        self.assertEqual(result["selected_action"], "WAIT")
        self.assertEqual(
            [call.kwargs["live_reasoning"] for call in mock_policy.call_args_list],
            [True, False]
        )
        self.assertTrue(controller._live_request_used)


if __name__ == "__main__":
    unittest.main(verbosity=2)
