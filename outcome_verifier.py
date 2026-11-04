"""Explicit verification states for recovery outcomes."""

from __future__ import annotations

from enum import Enum


class VerificationState(str, Enum):
    ACTION_EXECUTED = "ACTION_EXECUTED"
    OBSERVATION_REFRESHED = "OBSERVATION_REFRESHED"
    ENVIRONMENT_CHANGED = "ENVIRONMENT_CHANGED"
    PROGRESS_RESTORED = "PROGRESS_RESTORED"
    TASK_COMPLETED = "TASK_COMPLETED"
    RECOVERY_FAILED = "RECOVERY_FAILED"


def add_verification(states: tuple[VerificationState, ...], evidence: tuple[str, ...], state: VerificationState, detail: str) -> tuple[tuple[VerificationState, ...], tuple[str, ...]]:
    """Append a non-duplicated, auditable verification observation."""
    if state in states:
        return states, evidence
    return states + (state,), evidence + (detail,)


def progress_verified(states: tuple[VerificationState, ...]) -> bool:
    return VerificationState.PROGRESS_RESTORED in states or VerificationState.TASK_COMPLETED in states
