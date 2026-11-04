"""Regression checks for reproducible Milestone 8 experiments."""

from evaluation import EvaluationRunner


def run_evaluation_checks():
    runner = EvaluationRunner(trials_per_scenario=3)
    assert runner.configurations_are_equivalent()
    first, second = runner.run(), runner.run()
    assert first == second
    assert len(first) == 36
    assert all(item.reached_goal or not item.restored_progress for item in first)
    return runner.aggregate(first), runner.report(first)
