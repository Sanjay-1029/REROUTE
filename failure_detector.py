"""Structured, rule-based failure detection for REROUTE observations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from environment import Environment, MoveResult, Position


class FailureType(str, Enum):
    MOVEMENT_BLOCKED = "MOVEMENT_BLOCKED"
    ROUTE_OUTDATED = "ROUTE_OUTDATED"
    NO_PROGRESS = "NO_PROGRESS"
    UNEXPECTED_POSITION = "UNEXPECTED_POSITION"
    NO_ROUTE = "NO_ROUTE"
    GOAL_REACHED = "GOAL_REACHED"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class FailureRecord:
    failure_type: FailureType
    tick: int
    agent_position: Position
    expected_position: Position | None
    environment: dict[str, object]
    evidence: str
    severity: Severity
    temporary: bool


class FailureDetector:
    """Classifies observations and retains a history for pattern detection."""

    def __init__(self, no_progress_threshold: int = 2) -> None:
        self.no_progress_threshold = no_progress_threshold
        self.history: list[FailureRecord] = []
        self._stalled_position: Position | None = None
        self._consecutive_stalls = 0

    def observe_movement(self, environment: Environment, movement: MoveResult, expected: Position) -> FailureRecord | None:
        """Classify a movement result; successful expected movement is not a failure."""
        if not movement.moved:
            record = self._record(
                FailureType.MOVEMENT_BLOCKED, environment, expected,
                f"Movement was rejected: {movement.reason}.", Severity.ERROR,
                self._is_temporary_obstacle(environment, expected),
            )
            self._track_stall(environment, expected)
            return record
        if movement.current_position != expected:
            self._reset_stall()
            return self._record(
                FailureType.UNEXPECTED_POSITION, environment, expected,
                f"Movement reported success but reached {movement.current_position}, not {expected}.",
                Severity.ERROR, False,
            )
        self._reset_stall()
        return None

    def observe_route_outdated(self, environment: Environment, positions: Iterable[Position]) -> FailureRecord:
        blocked = tuple(positions)
        temporary = bool(blocked) and all(self._is_temporary_obstacle(environment, position) for position in blocked)
        return self._record(
            FailureType.ROUTE_OUTDATED, environment, blocked[0] if blocked else None,
            f"Remaining planned route is blocked at {blocked}.", Severity.ERROR, temporary,
        )

    def observe_no_route(self, environment: Environment) -> FailureRecord:
        return self._record(FailureType.NO_ROUTE, environment, None, "Pathfinding found no valid route.", Severity.ERROR, False)

    def observe_goal_reached(self, environment: Environment) -> FailureRecord:
        return self._record(FailureType.GOAL_REACHED, environment, environment.goal_position, "Agent position matches goal position.", Severity.INFO, False)

    def _track_stall(self, environment: Environment, expected: Position) -> None:
        position = environment.agent_position
        if position == self._stalled_position:
            self._consecutive_stalls += 1
        else:
            self._stalled_position, self._consecutive_stalls = position, 1
        if self._consecutive_stalls >= self.no_progress_threshold:
            self._record(
                FailureType.NO_PROGRESS, environment, expected,
                f"{self._consecutive_stalls} consecutive failed movements at {position}.", Severity.WARNING,
                self._is_temporary_obstacle(environment, expected),
            )

    def _record(self, failure_type: FailureType, environment: Environment, expected: Position | None, evidence: str, severity: Severity, temporary: bool) -> FailureRecord:
        state = dict(environment.get_state())
        state["expected_position_obstacle"] = environment.obstacle_info(expected) if expected is not None else None
        record = FailureRecord(failure_type, state["tick"], state["agent_position"], expected, state, evidence, severity, temporary)
        self.history.append(record)
        return record

    def _is_temporary_obstacle(self, environment: Environment, position: Position) -> bool:
        return environment.obstacle_info(position).get("temporary", False)

    def _reset_stall(self) -> None:
        self._stalled_position = None
        self._consecutive_stalls = 0
