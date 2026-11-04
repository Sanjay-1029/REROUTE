"""Regression checks for unseen-environment robustness experiments."""

from robustness import RobustnessRunner


def run_robustness_checks():
    runner = RobustnessRunner(trials_per_scenario=2)
    assert runner.configurations_are_equivalent()
    first, second = runner.run(), runner.run()
    assert first == second and len(first) == 32
    assert all(not item.restored_progress or item.reached_goal for item in first)
    assert all(runner.edge_cases().values())
    return runner.aggregate(first), runner.density_completion(first)
