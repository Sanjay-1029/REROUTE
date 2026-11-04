"""Uncertainty-aware recovery policy for diagnosed REROUTE failures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from diagnosis import DiagnosisCause, DiagnosisRecord
from environment import Environment
from outcome_verifier import VerificationState, add_verification


class RecoveryAction(str, Enum):
    WAIT = "WAIT"
    RETRY = "RETRY"
    GATHER_INFORMATION = "GATHER_INFORMATION"
    REPLAN = "REPLAN"
    REQUEST_HUMAN_HELP = "REQUEST_HUMAN_HELP"


@dataclass(frozen=True)
class RecoveryDecision:
    action: RecoveryAction
    reason: str


@dataclass(frozen=True)
class RecoveryRecord:
    diagnosis: DiagnosisRecord
    decision: RecoveryDecision
    attempt: int
    execution_result: str
    succeeded: bool
    ticks_spent: int = 0
    attempt_id: str = ""
    start_tick: int = 0
    end_tick: int = 0
    affected_route_nodes: tuple[tuple[int, int], ...] = ()
    observation_versions: tuple[int, ...] = ()
    verification_states: tuple[VerificationState, ...] = ()
    verification_evidence: tuple[str, ...] = ()


class RecoveryEngine:
    """Selects a bounded action from diagnosis, state, alternatives, and history."""

    def __init__(self, max_attempts: int = 3) -> None:
        self.max_attempts = max_attempts
        self.history: list[RecoveryRecord] = []
        self._next_attempt_id = 1

    def decide(self, diagnosis: DiagnosisRecord, environment: Environment, safe_alternative_route: bool) -> RecoveryDecision:
        attempts = sum(record.diagnosis.failure == diagnosis.failure for record in self.history)
        if len(self.history) >= self.max_attempts or attempts >= self.max_attempts:
            return RecoveryDecision(RecoveryAction.REQUEST_HUMAN_HELP, "Recovery limit reached or evidence is too uncertain for safe autonomy.")
        obstacle = environment.obstacle_info(diagnosis.failure.expected_position) if diagnosis.failure.expected_position else {}
        transient_move_failure = (
            "transient_failure" in diagnosis.failure.evidence
            and not obstacle.get("present", False)
            and safe_alternative_route
        )
        if transient_move_failure:
            return RecoveryDecision(RecoveryAction.RETRY, "Move failed transiently, but a fresh obstacle check is clear and the planned route remains safe.")
        if diagnosis.human_intervention_may_be_needed and diagnosis.confidence < 0.45:
            return RecoveryDecision(RecoveryAction.REQUEST_HUMAN_HELP, "Evidence is too uncertain and no safe, specific autonomous action is supported.")
        if diagnosis.cause == DiagnosisCause.TEMPORARY_OBSTACLE:
            if obstacle.get("present") and obstacle.get("remaining_duration") is not None:
                return RecoveryDecision(RecoveryAction.WAIT, "A timed obstacle is present; waiting can verify expiry.")
            return RecoveryDecision(RecoveryAction.RETRY, "Temporary condition is no longer observed; one verified retry is bounded and safe.")
        if diagnosis.cause in {DiagnosisCause.PERSISTENT_OBSTACLE, DiagnosisCause.REPEATED_FAILURE} and safe_alternative_route:
            return RecoveryDecision(RecoveryAction.REPLAN, "Evidence indicates the route is blocked and a safe alternative exists.")
        if diagnosis.cause == DiagnosisCause.STALE_ENVIRONMENT_INFORMATION:
            if attempts == 0:
                return RecoveryDecision(RecoveryAction.GATHER_INFORMATION, "A newer observation is needed before changing the plan.")
            if safe_alternative_route:
                return RecoveryDecision(RecoveryAction.REPLAN, "Fresh state still invalidates the route but supports an alternative.")
        if diagnosis.cause in {DiagnosisCause.INSUFFICIENT_INFORMATION, DiagnosisCause.UNKNOWN_CAUSE} and attempts == 0:
            return RecoveryDecision(RecoveryAction.GATHER_INFORMATION, "Uncertainty is high; collect a fresh state before another action.")
        if diagnosis.cause == DiagnosisCause.NO_VALID_ROUTE and diagnosis.confidence < 0.7:
            return RecoveryDecision(RecoveryAction.GATHER_INFORMATION, "Route evidence is inconclusive; verify the map once.")
        return RecoveryDecision(RecoveryAction.REQUEST_HUMAN_HELP, "No bounded autonomous action is supported by the available evidence.")

    def record(self, diagnosis: DiagnosisRecord, decision: RecoveryDecision, execution_result: str, succeeded: bool, ticks_spent: int = 0, start_tick: int = 0, end_tick: int = 0, affected_route_nodes: tuple[tuple[int, int], ...] = (), observation_versions: tuple[int, ...] = ()) -> RecoveryRecord:
        attempt = 1 + sum(record.diagnosis.failure == diagnosis.failure for record in self.history)
        attempt_id = f"recovery-{self._next_attempt_id:03d}"
        self._next_attempt_id += 1
        states, evidence = add_verification((), (), VerificationState.ACTION_EXECUTED, execution_result)
        if not succeeded:
            states, evidence = add_verification(states, evidence, VerificationState.RECOVERY_FAILED, execution_result)
        record = RecoveryRecord(diagnosis, decision, attempt, execution_result, succeeded, ticks_spent, attempt_id, start_tick, end_tick, affected_route_nodes, observation_versions, states, evidence)
        self.history.append(record)
        return record

    def verify(self, attempt_id: str, state: VerificationState, evidence: str) -> None:
        for index, record in enumerate(self.history):
            if record.attempt_id == attempt_id:
                states, details = add_verification(record.verification_states, record.verification_evidence, state, evidence)
                self.history[index] = RecoveryRecord(record.diagnosis, record.decision, record.attempt, record.execution_result, record.succeeded, record.ticks_spent, record.attempt_id, record.start_tick, record.end_tick, record.affected_route_nodes, record.observation_versions, states, details)
                return
