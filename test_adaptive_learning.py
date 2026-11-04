"""Tests for verified, diagnosis-aware adaptive outcome memory."""
import unittest
from types import SimpleNamespace

from adaptive_learning import AdaptiveOutcomeMemory
from outcome_verifier import VerificationState


def record(attempt, action, cause="TEMPORARY_OBSTACLE", states=(), succeeded=True):
    return SimpleNamespace(
        attempt_id=attempt,
        decision=SimpleNamespace(action=action),
        diagnosis=SimpleNamespace(cause=cause),
        verification_states=tuple(states),
        succeeded=succeeded,
    )


class AdaptiveLearningTests(unittest.TestCase):
    def test_unverified_attempt_is_not_learned(self):
        memory = AdaptiveOutcomeMemory()
        pending = record("r1", "REPLAN")
        self.assertEqual(memory.ingest([pending]), 0)
        self.assertEqual(memory.total_verified_outcomes, 0)

    def test_progress_restoration_is_learned_and_deduplicated(self):
        memory = AdaptiveOutcomeMemory()
        completed = record(
            "r1", "REPLAN",
            states=(VerificationState.ACTION_EXECUTED, VerificationState.PROGRESS_RESTORED),
        )
        self.assertEqual(memory.ingest([completed]), 1)
        self.assertEqual(memory.ingest([completed]), 0)
        stats = memory.snapshot("TEMPORARY_OBSTACLE")["action_statistics"]
        self.assertEqual(stats["REPLAN"], {"success": 1, "failure": 0})

    def test_gather_information_requires_actual_observation_refresh(self):
        memory = AdaptiveOutcomeMemory()
        failed_observation = record("r1", "GATHER_INFORMATION")
        refreshed = record(
            "r2", "GATHER_INFORMATION",
            states=(VerificationState.ACTION_EXECUTED, VerificationState.OBSERVATION_REFRESHED),
        )
        self.assertEqual(memory.ingest([failed_observation, refreshed]), 1)
        stats = memory.snapshot("TEMPORARY_OBSTACLE")["action_statistics"]
        self.assertEqual(stats["GATHER_INFORMATION"], {"success": 1, "failure": 0})

    def test_failure_is_learned_but_human_escalation_is_not(self):
        memory = AdaptiveOutcomeMemory()
        failed = record("r1", "RETRY", states=(VerificationState.RECOVERY_FAILED,), succeeded=False)
        human = record("r2", "REQUEST_HUMAN_HELP", states=(VerificationState.RECOVERY_FAILED,), succeeded=False)
        self.assertEqual(memory.ingest([failed, human]), 1)
        self.assertEqual(memory.snapshot("TEMPORARY_OBSTACLE")["action_statistics"]["RETRY"], {"success": 0, "failure": 1})

    def test_verified_success_and_failure_change_action_risk_estimate(self):
        from outcome_predictor import predict_outcome

        successful_memory = AdaptiveOutcomeMemory()
        failed_memory = AdaptiveOutcomeMemory()
        successes = [
            record(f"s{i}", "WAIT", states=(VerificationState.ENVIRONMENT_CHANGED,))
            for i in range(8)
        ]
        failures = [
            record(f"f{i}", "WAIT", states=(VerificationState.RECOVERY_FAILED,), succeeded=False)
            for i in range(8)
        ]
        successful_memory.ingest(successes)
        failed_memory.ingest(failures)

        state = {"temporary_obstacle": True}
        success_prediction = predict_outcome(
            "WAIT", state,
            successful_memory.snapshot("TEMPORARY_OBSTACLE")["action_statistics"],
        )
        failure_prediction = predict_outcome(
            "WAIT", state,
            failed_memory.snapshot("TEMPORARY_OBSTACLE")["action_statistics"],
        )
        self.assertEqual(success_prediction["risk_level"], "LOW")
        self.assertEqual(failure_prediction["risk_level"], "HIGH")

    def test_pipeline_includes_runtime_evidence_without_bypassing_safety(self):
        from unittest.mock import patch
        from decision_pipeline import run_decision_pipeline

        recommendation = {
            "recommended_action": "WAIT",
            "reason": "A temporary obstacle may clear.",
            "confidence": 0.8,
            "required_checks": ["Recheck the obstacle after waiting."],
            "alternative_action": "GATHER_INFORMATION",
            "visual_summary": "",
            "uncertainties": [],
        }
        context = {
            "temporary_obstacle": True,
            "adaptive_learning": {
                "source": "VERIFIED_RUNTIME_OUTCOMES",
                "diagnosis": "TEMPORARY_OBSTACLE",
                "observations": 3,
                "total_verified_outcomes": 3,
                "action_statistics": {"WAIT": {"success": 3, "failure": 0}},
            },
        }
        baseline = {"WAIT": {"success": 2, "failure": 1}}
        with patch("decision_pipeline._historical_action_statistics", return_value=baseline):
            result = run_decision_pipeline(context, recommendation_provider=lambda _: recommendation)

        wait_prediction = next(
            item["prediction"] for item in result["evaluated_actions"]
            if item["action"] == "WAIT"
        )
        self.assertEqual(wait_prediction["observations"], 6)
        self.assertEqual(result["adaptive_learning"]["verified_observations"], 3)
        # Adaptive statistics are evidence only: infeasible actions stay infeasible.
        self.assertFalse(next(
            item["prediction"]["feasible"] for item in result["evaluated_actions"]
            if item["action"] == "REPLAN"
        ))

    def test_diagnosis_snapshots_are_separated(self):
        memory = AdaptiveOutcomeMemory()
        memory.ingest([
            record("r1", "WAIT", "TEMPORARY_OBSTACLE", (VerificationState.ENVIRONMENT_CHANGED,)),
            record("r2", "RETRY", "REPEATED_FAILURE", (VerificationState.PROGRESS_RESTORED,)),
        ])
        self.assertIn("WAIT", memory.snapshot("TEMPORARY_OBSTACLE")["action_statistics"])
        self.assertNotIn("WAIT", memory.snapshot("REPEATED_FAILURE")["action_statistics"])


    def test_verified_memory_persists_across_instances(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            first = AdaptiveOutcomeMemory(storage_path=path, persistent=True)
            completed = record(
                "persist-1", "REPLAN",
                states=(VerificationState.ACTION_EXECUTED, VerificationState.PROGRESS_RESTORED),
            )
            self.assertEqual(first.ingest([completed]), 1)
            second = AdaptiveOutcomeMemory(storage_path=path, persistent=True)
            snapshot = second.snapshot("TEMPORARY_OBSTACLE")
            self.assertEqual(snapshot["action_statistics"]["REPLAN"], {"success": 1, "failure": 0})
            self.assertEqual(second.ingest([completed]), 0)

    def test_predictive_reasoner_scores_candidates_without_changing_feasibility(self):
        from predictive_reasoner import rank_candidate_actions

        candidates = [
            {"action": "REPLAN", "prediction": {
                "success_probability": 0.8, "feasible": True,
                "visual_uncertainty_count": 1,
                "visual_evidence_status": "VISUAL_EVIDENCE_ANALYZED",
            }},
            {"action": "RETRY", "prediction": {
                "success_probability": 0.9, "feasible": False,
                "visual_uncertainty_count": 1,
                "visual_evidence_status": "VISUAL_EVIDENCE_ANALYZED",
            }},
        ]
        ranked = rank_candidate_actions(candidates, {})
        retry = next(x for x in ranked if x["action"] == "RETRY")
        self.assertEqual(retry["prediction"]["predictive_assessment"]["expected_utility"], -1000.0)
        self.assertFalse(retry["prediction"]["feasible"])


if __name__ == "__main__":
    unittest.main()
