"""Tests for preserving complete decision history."""

import unittest

from simulation_controller import SimulationController
from agent import AgentEvent


class DecisionHistoryTests(unittest.TestCase):

    def test_recovering_event_preserves_complete_decision(self):
        controller = SimulationController()

        decision_details = {
            "status": "EVALUATED",
            "original_recommendation": {
                "recommended_action": "REPLAN",
                "confidence": 0.9
            },
            "selected_action": "REPLAN",
            "prediction": {
                "success_probability": 0.85,
                "risk_level": "LOW",
                "feasible": True
            },
            "evaluated_actions": [
                {"action": "REPLAN"},
                {"action": "WAIT"},
                {"action": "RETRY"},
                {"action": "GATHER_INFORMATION"},
                {"action": "REQUEST_HUMAN_HELP"}
            ],
            "uncertainties": ["Alternative route condition"]
        }

        event = AgentEvent(
            activity="RECOVERING",
            tick=5,
            state_version=3,
            payload={
                "state": {
                    "agent_position": (0, 0),
                    "goal_position": (4, 4),
                    "obstacles": [],
                    "width": 5,
                    "height": 5
                },
                "decision_source": "EXTERNAL_DECISION_PIPELINE",
                "recovery_action": "REPLAN",
                "decision_reason": "Decision pipeline recommendation",
                "decision_details": decision_details
            }
        )

        controller._apply_event(event)

        history = controller.snapshot.decision_history

        self.assertEqual(len(history), 1)

        record = history[0]

        self.assertEqual(
            record["decision_source"],
            "EXTERNAL_DECISION_PIPELINE"
        )

        self.assertEqual(record["recovery_action"], "REPLAN")

        self.assertEqual(
            record["decision_details"]["selected_action"],
            "REPLAN"
        )

        self.assertEqual(
            len(record["decision_details"]["evaluated_actions"]),
            5
        )

        self.assertEqual(
            record["decision_details"]["prediction"]["risk_level"],
            "LOW"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)