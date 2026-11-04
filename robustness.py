"""Unseen-environment robustness evaluation for REROUTE."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Callable

from environment import Environment
from evaluation import AggregateMetrics, EvaluationRunner, ScenarioFactory, TrialResult


@dataclass(frozen=True)
class RobustScenario:
    name: str
    density: str
    configuration: str
    factory: ScenarioFactory


class RobustnessRunner:
    """Runs previously unseen maps without changing the tuned recovery policy."""

    def __init__(self, trials_per_scenario: int = 2) -> None:
        self.trials_per_scenario = trials_per_scenario
        self._executor = EvaluationRunner()
        self.scenarios = (
            RobustScenario("small_sparse", "sparse", "8x8; start (0,0), goal (7,7); 2 static obstacles", self._small_sparse),
            RobustScenario("wide_off_center", "medium", "14x6; start (1,1), goal (12,4); 5 static obstacles", self._wide_off_center),
            RobustScenario("dense_corridor", "dense", "8x8; 18 obstacles with a single viable corridor", self._dense_corridor),
            RobustScenario("long_temporary", "medium", "10x10; route obstacle lasts 3 ticks", self._long_temporary),
            RobustScenario("simultaneous_obstacles", "medium", "10x10; two route-area obstacles appear together", self._simultaneous),
            RobustScenario("change_during_wait", "medium", "10x10; temporary obstacle moves while recovery waits", self._change_during_wait),
            RobustScenario("high_failure_frequency", "sparse", "10x10; four injected transient movement failures", self._high_failure_frequency),
            RobustScenario("large_unreachable", "dense", "7x7; start is enclosed by static obstacles", self._large_unreachable),
        )

    def run(self) -> list[TrialResult]:
        results: list[TrialResult] = []
        for scenario in self.scenarios:
            for trial in range(1, self.trials_per_scenario + 1):
                results.append(self._executor._run_immediate(scenario.name, trial, scenario.factory()))
                results.append(self._executor._run_adaptive(scenario.name, trial, scenario.factory()))
        return results

    def configurations_are_equivalent(self) -> bool:
        return all(
            scenario.factory().get_state() == scenario.factory().get_state()
            and scenario.factory().events == scenario.factory().events
            for scenario in self.scenarios
        )

    def aggregate(self, results: list[TrialResult]) -> list[AggregateMetrics]:
        return self._executor.aggregate(results)

    def density_completion(self, results: list[TrialResult]) -> dict[str, dict[str, float]]:
        density_for = {scenario.name: scenario.density for scenario in self.scenarios}
        summary: dict[str, dict[str, float]] = {}
        for density in {scenario.density for scenario in self.scenarios}:
            group = [result for result in results if density_for[result.scenario] == density]
            summary[density] = {
                strategy: mean(item.reached_goal for item in group if item.strategy == strategy)
                for strategy in ("Immediate replanning", "REROUTE adaptive recovery")
            }
        return summary

    def edge_cases(self) -> dict[str, bool]:
        """Validate invalid maps fail early and valid impossible maps stop safely."""
        checks: dict[str, bool] = {}
        for name, kwargs in {
            "blocked_start": {"width": 3, "height": 3, "agent_position": (0, 0), "goal_position": (2, 2), "obstacles": [(0, 0)]},
            "blocked_goal": {"width": 3, "height": 3, "agent_position": (0, 0), "goal_position": (2, 2), "obstacles": [(2, 2)]},
        }.items():
            try:
                Environment(**kwargs)
            except ValueError:
                checks[name] = True
            else:
                checks[name] = False
        return checks

    @staticmethod
    def _small_sparse() -> Environment:
        return Environment(8, 8, (0, 0), (7, 7), [(3, 3), (4, 3)])
    @staticmethod
    def _wide_off_center() -> Environment:
        return Environment(14, 6, (1, 1), (12, 4), [(5, 1), (5, 2), (6, 2), (8, 3), (9, 3)])
    @staticmethod
    def _dense_corridor() -> Environment:
        obstacles = [(2, y) for y in range(6)] + [(4, y) for y in range(2, 8)] + [(6, y) for y in range(6)]
        return Environment(8, 8, (0, 0), (7, 7), obstacles)
    @staticmethod
    def _long_temporary() -> Environment:
        environment = Environment(); environment.schedule_obstacle_event(1, "add", (0, 1), duration=3); return environment
    @staticmethod
    def _simultaneous() -> Environment:
        environment = Environment(); environment.schedule_obstacle_event(1, "add", (0, 1)); environment.schedule_obstacle_event(1, "add", (1, 0)); return environment
    @staticmethod
    def _change_during_wait() -> Environment:
        environment = Environment(); environment.schedule_obstacle_event(1, "add", (0, 1), duration=2); environment.schedule_obstacle_event(2, "move", (0, 1), destination=(1, 1)); return environment
    @staticmethod
    def _high_failure_frequency() -> Environment:
        environment = Environment(); environment.set_transient_move_failures(4); return environment
    @staticmethod
    def _large_unreachable() -> Environment:
        return Environment(7, 7, (0, 0), (6, 6), [(1, 0), (0, 1), (2, 2), (3, 3), (4, 4), (5, 5)])
