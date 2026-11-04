"""Offline regression tests. These tests never call the Meta API."""

import unittest

from decision_pipeline import run_decision_pipeline
from decision_selector import select_best_action
from environment import Environment
from outcome_predictor import predict_outcome
from recommendation_validator import validate_recommendation
from scenarios import run_dynamic_scenarios
from simulation_controller import mock_recommendation
from visual_evidence import assess_visual_evidence

class DecisionSafetyTests(unittest.TestCase):

    def test_visual_uncertainty_blocks_unsafe_retry(self):
        from outcome_predictor import predict_outcome

        state = {
            "path_blocked": False,
            "visual_evidence": {
                "evidence_status": "VISUAL_EVIDENCE_ANALYZED",
                "uncertainties": ["Obstacle position is unknown."],
            },
        }

        result = predict_outcome("RETRY", state, {
            "RETRY": {"success": 8, "failure": 2}
        })

        self.assertFalse(result["feasible"])
        self.assertEqual(result["risk_level"], "HIGH")


    def test_replan_requires_confirmed_safe_alternative(self):
        from outcome_predictor import predict_outcome

        state = {
            "safe_alternative_route": False,
            "visual_evidence": {
                "evidence_status": "VISUAL_EVIDENCE_ANALYZED",
                "uncertainties": ["Alternative route is unknown."],
            },
        }

        result = predict_outcome("REPLAN", state, {
            "REPLAN": {"success": 8, "failure": 2}
        })

        self.assertFalse(result["feasible"])
        self.assertEqual(result["risk_level"], "HIGH")


    def test_gather_information_remains_feasible_with_uncertainty(self):
        from outcome_predictor import predict_outcome

        state = {
            "visual_evidence": {
                "evidence_status": "IMAGE_AVAILABLE_NOT_ANALYZED",
                "uncertainties": ["Environment is not yet understood."],
            },
        }

        result = predict_outcome("GATHER_INFORMATION", state, {
            "GATHER_INFORMATION": {"success": 8, "failure": 2}
        })

        self.assertTrue(result["feasible"])
        self.assertIn("reduce uncertainty", result["uncertainty_reason"])


    def test_human_help_remains_available_with_visual_uncertainty(self):
        from outcome_predictor import predict_outcome

        state = {
            "visual_evidence": {
                "evidence_status": "NO_VERIFIED_VISUAL_INPUT",
                "uncertainties": ["Environment is unknown."],
            },
        }

        result = predict_outcome("REQUEST_HUMAN_HELP", state, {
            "REQUEST_HUMAN_HELP": {"success": 8, "failure": 2}
        })

        self.assertTrue(result["feasible"])
        self.assertIn("Human assistance", result["uncertainty_reason"])

    def test_image_ready_is_not_treated_as_analyzed(self):
        from visual_evidence import assess_visual_evidence

        context = {
            "perception_status": "IMAGE_READY_FOR_MUSE_SPARK",
            "visual_observations": [],
        }

        recommendation = {
            "uncertainties": ["Obstacle position is unknown."]
        }

        result = assess_visual_evidence(context, recommendation)

        self.assertTrue(result["image_available"])
        self.assertFalse(result["analysis_completed"])
        self.assertEqual(
            result["evidence_status"],
            "IMAGE_AVAILABLE_NOT_ANALYZED",
        )

    def test_visual_uncertainty_reaches_outcome_prediction(self):
        from outcome_predictor import predict_outcome

        environment_state = {
            "safe_alternative_route": True,
            "visual_evidence": {
                "evidence_status": "VISUAL_EVIDENCE_ANALYZED",
                "uncertainties": [
                    "Exact obstacle position is unknown."
                ],
            },
        }

        statistics = {
            "REPLAN": {
                "success": 8,
                "failure": 2,
            }
        }

        result = predict_outcome(
            "REPLAN",
            environment_state,
            statistics,
        )

        self.assertEqual(
            result["visual_evidence_status"],
            "VISUAL_EVIDENCE_ANALYZED",
        )
        self.assertEqual(result["visual_uncertainty_count"], 1)
        self.assertIn("unresolved uncertainties", result["expected_result"])


    def test_visual_evidence_assessment(self):
        context = {
            "perception_status": "IMAGE_READY_FOR_MUSE_SPARK",
            "visual_observations": [
                "An object appears to block the hallway."
            ]
        }

        recommendation = {
            "uncertainties": [
                "Exact object position is unknown."
            ]
        }

        result = assess_visual_evidence(context, recommendation)

        self.assertTrue(result["image_available"])
        self.assertEqual(result["observation_count"], 1)
        self.assertEqual(len(result["uncertainties"]), 1)
        self.assertFalse(result["navigation_coordinates_verified"])

    def test_replan_rejected_without_safe_alternative(self):
        rec = {
            "recommended_action": "REPLAN",
            "reason": "Blocked",
            "confidence": 0.9,
            "required_checks": ["Check route"],
            "alternative_action": "GATHER_INFORMATION"
        }

        ok, message = validate_recommendation(
            rec,
            {
                "safe_alternative_route": False,
                "obstacles_known": True
            }
        )

        self.assertFalse(ok)
        self.assertIn("No confirmed safe alternative", message)

    def test_retry_rejected_while_path_blocked(self):
        rec = {
            "recommended_action": "RETRY",
            "reason": "Try again",
            "confidence": 0.7,
            "required_checks": ["Check obstruction"],
            "alternative_action": "WAIT"
        }

        ok, message = validate_recommendation(
            rec,
            {"path_blocked": True}
        )

        self.assertFalse(ok)
        self.assertIn("Cannot retry", message)

    def test_selector_never_overrides_high_or_unknown_risk(self):
        items = [
            {
                "action": "REPLAN",
                "prediction": {
                    "success_probability": 0.99,
                    "lower_confidence_bound": 0.9,
                    "risk_level": "HIGH",
                    "feasible": True
                }
            },
            {
                "action": "WAIT",
                "prediction": {
                    "success_probability": 0.5,
                    "risk_level": "UNKNOWN",
                    "feasible": True
                }
            },
        ]

        self.assertIsNone(
            select_best_action(items, recommended_action="REPLAN")
        )

    # NEW TEST 1:
    # A safer action must override a weaker model recommendation.
    def test_safer_action_overrides_weaker_recommendation(self):
        items = [
            {
                "action": "REPLAN",
                "prediction": {
                    "success_probability": 0.95,
                    "lower_confidence_bound": 0.85,
                    "risk_level": "LOW",
                    "feasible": True
                }
            },
            {
                "action": "WAIT",
                "prediction": {
                    "success_probability": 0.60,
                    "lower_confidence_bound": 0.40,
                    "risk_level": "MEDIUM",
                    "feasible": True
                }
            }
        ]

        selected = select_best_action(
            items,
            recommended_action="WAIT"
        )

        self.assertEqual(selected["action"], "REPLAN")

    # NEW TEST 2:
    # The model recommendation breaks a tie when evidence is equal.
    def test_recommendation_breaks_equal_evidence_tie(self):
        items = [
            {
                "action": "REPLAN",
                "prediction": {
                    "success_probability": 0.90,
                    "lower_confidence_bound": 0.75,
                    "risk_level": "LOW",
                    "feasible": True
                }
            },
            {
                "action": "WAIT",
                "prediction": {
                    "success_probability": 0.90,
                    "lower_confidence_bound": 0.75,
                    "risk_level": "LOW",
                    "feasible": True
                }
            }
        ]

        selected = select_best_action(
            items,
            recommended_action="WAIT"
        )

        self.assertEqual(selected["action"], "WAIT")

    def test_predictor_marks_blocked_retry_infeasible(self):
        prediction = predict_outcome(
            "RETRY",
            {"path_blocked": True},
            {"RETRY": {"success": 8, "failure": 0}}
        )

        self.assertFalse(prediction["feasible"])
        self.assertEqual(prediction["risk_level"], "HIGH")
        self.assertEqual(prediction["success_probability"], 0.0)

    def test_pipeline_uses_model_preference_only_if_safe(self):
        context = {
            "safe_alternative_route": True,
            "obstacles_known": True,
            "path_blocked": True,
            "environment_state": {
                "agent_position": [0, 0],
                "goal_position": [4, 4],
                "obstacles": []
            }
        }

        result = run_decision_pipeline(
            context,
            recommendation_provider=lambda _: mock_recommendation(context)
        )

        self.assertEqual(result["status"], "EVALUATED")
        self.assertIn(result["selected_action"], {"REPLAN", "WAIT"})
        self.assertTrue(result["prediction"]["feasible"])

    def test_pipeline_preserves_visual_evidence_and_uncertainties(self):
        context = {
            "safe_alternative_route": True,
            "obstacles_known": True,
            "path_blocked": True,
            "visual_observations": [
                "An object appears to block the hallway."
            ],
            "visual_detection_count": 1,
            "perception_status": "IMAGE_ANALYZED",
            "environment_state": {
                "agent_position": [0, 0],
                "goal_position": [4, 4],
                "obstacles": []
            }
        }

        recommendation = {
            "recommended_action": "REPLAN",
            "reason": "A possible obstruction is visible.",
            "confidence": 0.9,
            "required_checks": ["Verify alternative route."],
            "alternative_action": "GATHER_INFORMATION",
            "visual_summary": "An object appears in the hallway.",
            "uncertainties": [
                "Exact object position is unknown."
            ]
        }

        result = run_decision_pipeline(
            context,
            recommendation_provider=lambda _: recommendation
        )

        self.assertEqual(result["status"], "EVALUATED")
        self.assertEqual(
            result["visual_summary"],
            recommendation["visual_summary"]
        )
        self.assertEqual(
            result["uncertainties"],
            recommendation["uncertainties"]
        )
        self.assertIn("evaluated_actions", result)    


class AutonomousLoopTests(unittest.TestCase):

    def test_three_dynamic_scenarios_reach_goal_offline(self):
        results = run_dynamic_scenarios(
            logger=lambda _: None,
            decision_policy=lambda c: run_decision_pipeline(
                c,
                recommendation_provider=mock_recommendation
            )
        )

        self.assertEqual(len(results), 3)
        self.assertTrue(
            all(item.outcome.reached_goal for item in results)
        )

    def test_plain_agent_reaches_goal_and_verifies_position(self):
        env = Environment()

        from agent import AutonomousAgent

        result = AutonomousAgent(
            env,
            logger=lambda _: None,
            recovery_mode="m11"
        ).run()

        self.assertTrue(result.reached_goal)
        self.assertEqual(env.agent_position, env.goal_position)


if __name__ == "__main__":
    unittest.main(verbosity=2)