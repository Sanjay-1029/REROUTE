"""Evidence-based cause assessment for REROUTE failure records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from environment import Environment
from failure_detector import FailureRecord, FailureType


class DiagnosisCause(str, Enum):
    TEMPORARY_OBSTACLE = "TEMPORARY_OBSTACLE"
    PERSISTENT_OBSTACLE = "PERSISTENT_OBSTACLE"
    INVALID_MOVEMENT = "INVALID_MOVEMENT"
    STALE_ENVIRONMENT_INFORMATION = "STALE_ENVIRONMENT_INFORMATION"
    NO_VALID_ROUTE = "NO_VALID_ROUTE"
    REPEATED_FAILURE = "REPEATED_FAILURE"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"
    UNKNOWN_CAUSE = "UNKNOWN_CAUSE"


@dataclass(frozen=True)
class DiagnosisRecord:
    failure: FailureRecord
    cause: DiagnosisCause
    confidence: float
    supporting_evidence: tuple[str, ...]
    missing_or_contradicting_evidence: tuple[str, ...]
    recommended_next_investigation: str
    human_intervention_may_be_needed: bool


class FailureDiagnosis:
    """Maps failure evidence to cautious, explainable likely causes."""

    def __init__(self) -> None:
        self.history: list[DiagnosisRecord] = []
        self._by_failure: dict[tuple[object, ...], DiagnosisRecord] = {}

    def diagnose(
        self,
        failure: FailureRecord,
        environment: Environment | None = None,
        movement_history: Iterable[object] = (),
        previous_failures: Iterable[FailureRecord] = (),
    ) -> DiagnosisRecord:
        """Diagnose once per original failure record using only supplied evidence."""
        key = self._key(failure)
        if key in self._by_failure:
            return self._by_failure[key]
        current = environment.get_state() if environment else None
        obstacle = environment.obstacle_info(failure.expected_position) if environment and failure.expected_position else failure.environment.get("expected_position_obstacle")
        prior = list(previous_failures)
        diagnosis = self._classify(failure, current, obstacle, tuple(movement_history), prior)
        self.history.append(diagnosis)
        self._by_failure[key] = diagnosis
        return diagnosis

    def _classify(self, failure, current, obstacle, movement_history, previous_failures) -> DiagnosisRecord:
        if not failure.environment or "agent_position" not in failure.environment:
            return self._make(failure, DiagnosisCause.INSUFFICIENT_INFORMATION, 0.20,
                ("Failure record lacks an environment snapshot.",), ("Position, obstacle, and route context are missing.",),
                "Capture a fresh state snapshot and repeat the observation.", True)
        repeated = sum(record.failure_type == failure.failure_type and record.agent_position == failure.agent_position for record in previous_failures)
        if failure.failure_type == FailureType.NO_PROGRESS or repeated >= 2:
            return self._make(failure, DiagnosisCause.REPEATED_FAILURE, 0.78,
                (failure.evidence, f"{max(repeated, 2)} matching failures at {failure.agent_position}."),
                ("Root cause of the repeated failure is not established.",), "Inspect the blocking condition and route assumptions.", False)
        if failure.failure_type == FailureType.NO_ROUTE:
            return self._make(failure, DiagnosisCause.NO_VALID_ROUTE, 0.82,
                (failure.evidence, "Pathfinding exhausted the current grid state."),
                ("Future environmental changes could make a route available.",), "Verify obstacle observations and map boundaries.", True)
        text = failure.evidence.lower()
        if "invalid_direction" in text or "outside_grid" in text:
            return self._make(failure, DiagnosisCause.INVALID_MOVEMENT, 0.84,
                (failure.evidence,), ("The intended controller command is not otherwise recorded.",),
                "Validate action selection against legal grid directions.", False)
        recorded_obstacles = set(failure.environment.get("obstacles", ()))
        if current and failure.expected_position and failure.expected_position in set(current.get("obstacles", ())):
            old_version = failure.environment.get("state_version", -1)
            if failure.expected_position not in recorded_obstacles and current.get("state_version", -1) > old_version:
                return self._make(failure, DiagnosisCause.STALE_ENVIRONMENT_INFORMATION, 0.66,
                    ("A newer snapshot contains an obstacle absent from the recorded snapshot.",),
                    ("The source and timing of the environmental change are not known.",), "Refresh the world model before selecting another action.", False)
        if obstacle and obstacle.get("present") and obstacle.get("temporary"):
            return self._make(failure, DiagnosisCause.TEMPORARY_OBSTACLE, 0.76,
                (failure.evidence, f"Dynamic obstacle has remaining duration {obstacle.get('remaining_duration')}."),
                ("Its actual removal time has not yet been observed.",), "Observe future ticks to confirm obstacle expiry.", False)
        if obstacle and obstacle.get("present"):
            return self._make(failure, DiagnosisCause.PERSISTENT_OBSTACLE, 0.74,
                (failure.evidence, f"{obstacle.get('kind')} obstacle is present at the expected position."),
                ("Persistence is inferred from current state, not long-term observation.",), "Observe additional ticks or inspect obstacle lifecycle metadata.", False)
        if failure.failure_type == FailureType.UNEXPECTED_POSITION:
            return self._make(failure, DiagnosisCause.UNKNOWN_CAUSE, 0.28,
                (failure.evidence,), ("No motion, sensor, or external-event evidence explains the mismatch.",),
                "Collect an independent position observation and action trace.", True)
        return self._make(failure, DiagnosisCause.INSUFFICIENT_INFORMATION, 0.30,
            (failure.evidence,), ("Available evidence does not distinguish obstacle, action, or state causes.",),
            "Gather another state snapshot and movement observation.", True)

    @staticmethod
    def _make(failure, cause, confidence, supporting, missing, next_step, human) -> DiagnosisRecord:
        return DiagnosisRecord(failure, cause, max(0.0, min(1.0, confidence)), tuple(supporting), tuple(missing), next_step, human)

    @staticmethod
    def _key(failure: FailureRecord) -> tuple[object, ...]:
        return (failure.failure_type, failure.tick, failure.agent_position, failure.expected_position, failure.evidence, repr(failure.environment))
