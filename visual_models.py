"""Read-only models consumed by the live dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActivityItem:
    activity: str
    tick: int
    message: str


@dataclass
class DashboardSnapshot:
    scenario: str = "Default navigation"
    status: str = "READY"
    state: dict[str, Any] = field(default_factory=dict)
    start_position: tuple[int, int] | None = None
    route: tuple[tuple[int, int], ...] = ()
    alternative_route: tuple[tuple[int, int], ...] = ()
    visited: set[tuple[int, int]] = field(default_factory=set)
    activity: list[ActivityItem] = field(default_factory=list)
    reasoning: dict[str, str] = field(default_factory=dict)

    # Complete recovery decision and prediction history.
    decision_history: list[dict[str, Any]] = field(default_factory=list)

    nodes_explored: int = 0
    failed_movements: int = 0
    retries: int = 0
    replans: int = 0
    recovery_attempts: int = 0
    reached_goal: bool = False
    safe_stop_reason: str = ""