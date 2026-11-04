"""Repeatable baseline-versus-REROUTE outcome evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Callable

from agent import AutonomousAgent
from environment import Environment
from pathfinder import astar_search
from recovery import RecoveryAction


@dataclass(frozen=True)
class TrialResult:
    strategy: str
    scenario: str
    trial: int
    reached_goal: bool
    restored_progress: bool
    ticks: int
    path_length: int
    failed_actions: int
    recovery_attempts: int
    recovery_successes: int
    recovery_ticks: int
    replans: int
    human_requests: int
    recovery_outcomes: tuple[tuple[str, bool], ...] = ()


@dataclass(frozen=True)
class AggregateMetrics:
    strategy: str
    trials: int
    completion_rate: float
    recovery_success_rate: float
    average_completion_ticks: float
    average_path_length: float
    failed_actions: int
    replans: int
    human_requests: int
    unsuccessful_tasks: int
    average_recovery_ticks: float


ScenarioFactory = Callable[[], Environment]


class EvaluationRunner:
    """Runs equivalent deterministic environments for both navigation strategies."""

    def __init__(self, trials_per_scenario: int = 3) -> None:
        self.trials_per_scenario = trials_per_scenario
        self.scenarios: dict[str, ScenarioFactory] = {
            "temporary_obstacle": self._temporary_obstacle,
            "persistent_obstacle": self._persistent_obstacle,
            "moving_obstacle": self._moving_obstacle,
            "repeated_movement_failures": self._repeated_failures,
            "uncertain_information": self._uncertain_information,
            "unreachable_goal": self._unreachable_goal,
        }

    def run(self) -> list[TrialResult]:
        results: list[TrialResult] = []

        for name, factory in self.scenarios.items():
            for trial in range(1, self.trials_per_scenario + 1):
                results.append(
                    self._run_immediate(name, trial, factory())
                )
                results.append(
                    self._run_adaptive(name, trial, factory())
                )

        return results

    def configurations_are_equivalent(self) -> bool:
        """Verify factory-created starting states match before assigning strategies."""

        for factory in self.scenarios.values():
            first, second = factory(), factory()

            if (
                first.get_state() != second.get_state()
                or first.events != second.events
            ):
                return False

        return True

    def aggregate(self, results: list[TrialResult]) -> list[AggregateMetrics]:
        output: list[AggregateMetrics] = []

        for strategy in (
            "Immediate replanning",
            "REROUTE adaptive recovery"
        ):
            group = [
                item for item in results
                if item.strategy == strategy
            ]

            completed = [
                item for item in group
                if item.reached_goal
            ]

            recovery_attempts = sum(
                item.recovery_attempts for item in group
            )

            output.append(
                AggregateMetrics(
                    strategy,
                    len(group),
                    len(completed) / len(group) if group else 0.0,
                    (
                        sum(item.recovery_successes for item in group)
                        / recovery_attempts
                        if recovery_attempts else 0.0
                    ),
                    mean(item.ticks for item in completed)
                    if completed else 0.0,
                    mean(item.path_length for item in completed)
                    if completed else 0.0,
                    sum(item.failed_actions for item in group),
                    sum(item.replans for item in group),
                    sum(item.human_requests for item in group),
                    len(group) - len(completed),
                    (
                        sum(item.recovery_ticks for item in group)
                        / recovery_attempts
                        if recovery_attempts else 0.0
                    ),
                )
            )

        return output

    def report(self, results: list[TrialResult]) -> str:
        metrics = self.aggregate(results)

        lines = [
            "| Strategy | Completion | Recovery success | Avg completion ticks | Avg recovery ticks | Avg path length | Failed actions | Replans | Human requests | Unsuccessful |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        ]

        for item in metrics:
            lines.append(
                f"| {item.strategy} | "
                f"{item.completion_rate:.0%} | "
                f"{item.recovery_success_rate:.0%} | "
                f"{item.average_completion_ticks:.1f} | "
                f"{item.average_recovery_ticks:.1f} | "
                f"{item.average_path_length:.1f} | "
                f"{item.failed_actions} | "
                f"{item.replans} | "
                f"{item.human_requests} | "
                f"{item.unsuccessful_tasks} |"
            )

        baseline, adaptive = metrics

        difference = (
            adaptive.completion_rate - baseline.completion_rate
        ) * 100

        lines.append(
            f"\nCompletion-rate difference "
            f"(adaptive - immediate): {difference:+.1f} percentage points."
        )

        return "\n".join(lines)

    def _run_adaptive(
        self,
        scenario: str,
        trial: int,
        environment: Environment
    ) -> TrialResult:

        outcome = AutonomousAgent(
            environment,
            logger=lambda _: None
        ).run()

        recoveries = outcome.recoveries

        # Preserve individual action outcomes for future prediction.
        recovery_outcomes = tuple(
            (
                record.decision.action.value,
                record.succeeded
            )
            for record in recoveries
        )

        return TrialResult(
            "REROUTE adaptive recovery",
            scenario,
            trial,
            outcome.reached_goal,
            outcome.reached_goal and any(
                record.succeeded for record in recoveries
            ),
            environment.tick,
            outcome.successful_moves,
            outcome.failed_moves,
            len(recoveries),
            sum(record.succeeded for record in recoveries),
            sum(record.ticks_spent for record in recoveries),
            sum(
                record.decision.action == RecoveryAction.REPLAN
                for record in recoveries
            ),
            sum(
                record.decision.action == RecoveryAction.REQUEST_HUMAN_HELP
                for record in recoveries
            ),
            recovery_outcomes
        )

    def _run_immediate(
        self,
        scenario: str,
        trial: int,
        environment: Environment
    ) -> TrialResult:

        successful_moves = 0
        failed_actions = 0
        replans = 0
        recovery_attempts = 0
        recovery_successes = 0

        limit = environment.width * environment.height * 3

        while successful_moves + failed_actions < limit:

            state = environment.get_state()

            route = astar_search(
                state["agent_position"],
                state["goal_position"],
                state["obstacles"],
                state["width"],
                state["height"]
            ).path

            if not route:
                break

            restarted = False

            for next_position in route[1:]:

                environment.advance()

                if tuple(next_position) in set(
                    environment.get_state()["obstacles"]
                ):
                    replans += 1
                    recovery_attempts += 1

                    alternative = astar_search(
                        environment.agent_position,
                        environment.goal_position,
                        environment.obstacles,
                        environment.width,
                        environment.height
                    ).path

                    recovery_successes += bool(alternative)
                    restarted = True
                    break

                direction = _direction(
                    environment.agent_position,
                    tuple(next_position)
                )

                result = environment.move(direction)

                if not result.moved:

                    failed_actions += 1
                    replans += 1
                    recovery_attempts += 1

                    alternative = astar_search(
                        environment.agent_position,
                        environment.goal_position,
                        environment.obstacles,
                        environment.width,
                        environment.height
                    ).path

                    recovery_successes += bool(alternative)
                    restarted = True
                    break

                successful_moves += 1

                if environment.agent_position == environment.goal_position:

                    return TrialResult(
                        "Immediate replanning",
                        scenario,
                        trial,
                        True,
                        recovery_successes > 0,
                        environment.tick,
                        successful_moves,
                        failed_actions,
                        recovery_attempts,
                        recovery_successes,
                        0,
                        replans,
                        0
                    )

            if not restarted:
                break

        return TrialResult(
            "Immediate replanning",
            scenario,
            trial,
            False,
            False,
            environment.tick,
            successful_moves,
            failed_actions,
            recovery_attempts,
            recovery_successes,
            0,
            replans,
            0
        )

    @staticmethod
    def _temporary_obstacle() -> Environment:
        environment = Environment()
        environment.schedule_obstacle_event(
            1, "add", (0, 1), duration=1
        )
        return environment

    @staticmethod
    def _persistent_obstacle() -> Environment:
        environment = Environment()
        environment.schedule_obstacle_event(
            1, "add", (0, 1)
        )
        return environment

    @staticmethod
    def _moving_obstacle() -> Environment:
        environment = Environment()
        environment.add_dynamic_obstacle((1, 1))
        environment.schedule_obstacle_event(
            1, "move", (1, 1), destination=(0, 1)
        )
        return environment

    @staticmethod
    def _repeated_failures() -> Environment:
        environment = Environment()
        environment.set_transient_move_failures(2)
        return environment

    @staticmethod
    def _uncertain_information() -> Environment:
        environment = Environment()
        environment.set_transient_move_failures(1)
        return environment

    @staticmethod
    def _unreachable_goal() -> Environment:
        return Environment(
            width=3,
            height=3,
            agent_position=(0, 0),
            goal_position=(2, 2),
            obstacles=[(1, 0), (0, 1)]
        )


def _direction(
    current: tuple[int, int],
    target: tuple[int, int]
) -> str:

    return {
        (0, -1): "up",
        (0, 1): "down",
        (-1, 0): "left",
        (1, 0): "right"
    }[
        (
            target[0] - current[0],
            target[1] - current[1]
        )
    ]