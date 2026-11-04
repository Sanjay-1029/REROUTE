"""Deterministic classification checks for every Milestone 5 failure type."""

from __future__ import annotations

from agent import AutonomousAgent
from environment import Environment, MoveResult
from failure_detector import FailureDetector, FailureType
from scenarios import run_dynamic_scenarios


def run_failure_scenarios() -> dict[str, list[FailureType]]:
    """Return detected types for compact, reproducible failure-classification tests."""
    results: dict[str, list[FailureType]] = {}

    # Normal movement produces no failure record.
    environment = Environment(width=3, height=3, agent_position=(0, 0), goal_position=(2, 2), obstacles=[])
    detector = FailureDetector()
    movement = environment.move("right")
    assert detector.observe_movement(environment, movement, (1, 0)) is None
    results["normal_movement"] = [record.failure_type for record in detector.history]

    environment = Environment(width=3, height=3, agent_position=(0, 0), goal_position=(2, 2), obstacles=[(1, 0)])
    detector = FailureDetector()
    for _ in range(2):
        detector.observe_movement(environment, environment.move("right"), (1, 0))
    results["movement_blocked_and_no_progress"] = [record.failure_type for record in detector.history]

    dynamic = run_dynamic_scenarios(logger=lambda _: None)[0]
    results["route_outdated"] = [record.failure_type for record in dynamic.outcome.failures]

    environment = Environment(width=3, height=3, agent_position=(0, 0), goal_position=(2, 2), obstacles=[])
    environment.agent_position = (1, 0)  # Controlled sensor/state mismatch observation.
    detector = FailureDetector()
    detector.observe_movement(environment, MoveResult(True, "moved", (0, 0), (1, 0)), (0, 1))
    results["unexpected_position"] = [record.failure_type for record in detector.history]

    trapped = Environment(width=3, height=3, agent_position=(0, 0), goal_position=(2, 2), obstacles=[(1, 0), (0, 1)])
    results["no_route"] = [record.failure_type for record in AutonomousAgent(trapped, logger=lambda _: None).run().failures]

    goal = Environment(width=2, height=1, agent_position=(0, 0), goal_position=(1, 0), obstacles=[])
    results["goal_reached"] = [record.failure_type for record in AutonomousAgent(goal, logger=lambda _: None).run().failures]
    return results
