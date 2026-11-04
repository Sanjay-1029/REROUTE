"""REROUTE dynamic scenarios with visual perception integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

from agent import AutonomousAgent, AgentRunResult
from environment import Environment


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    initial_route: list[list[int]]
    outcome: AgentRunResult
    changes: list[str]
    temporary_obstacle_cleared: bool = False


def run_dynamic_scenarios(
    logger=print,
    vision_engine=None,
    image_provider=None,
    decision_policy=None,
) -> list[ScenarioResult]:

    results = []

    # Scenario 1: New obstacle appears
    environment = Environment()

    environment.schedule_obstacle_event(
        1, "add", (0, 1)
    )

    results.append(
        _run(
            "Obstacle appears on route",
            environment,
            logger,
            vision_engine,
            image_provider,
            decision_policy,
        )
    )

    # Scenario 2: Temporary obstacle
    environment = Environment()

    environment.schedule_obstacle_event(
        1, "add", (0, 1), duration=1
    )

    result = _run(
        "Temporary obstacle blocks route",
        environment,
        logger,
        vision_engine,
        image_provider,
        decision_policy,
    )

    environment.advance()

    results.append(
        ScenarioResult(
            result.name,
            result.initial_route,
            result.outcome,
            environment.events,
            (0, 1) not in set(map(tuple, environment.obstacles)),
        )
    )

    # Scenario 3: Moving obstacle
    environment = Environment()

    environment.add_dynamic_obstacle((1, 1))

    environment.schedule_obstacle_event(
        1,
        "move",
        (1, 1),
        destination=(0, 1),
    )

    results.append(
        _run(
            "Moving obstacle changes route",
            environment,
            logger,
            vision_engine,
            image_provider,
            decision_policy,
        )
    )

    return results


def _run(
    name: str,
    environment: Environment,
    logger,
    vision_engine,
    image_provider,
    decision_policy,
) -> ScenarioResult:

    logger(f"\n=== Scenario: {name} ===")

    agent = AutonomousAgent(
        environment,
        algorithm="astar",
        logger=logger,
        recovery_enabled=True,
        recovery_mode="m11",
        vision_engine=vision_engine,
        image_provider=image_provider,
        decision_policy=decision_policy,
    )

    initial = agent.plan_route().path

    outcome = agent.run()

    return ScenarioResult(
        name,
        initial or [],
        outcome,
        list(environment.events),
    )