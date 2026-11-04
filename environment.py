"""Grid-world simulation used by the REROUTE agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

Position = tuple[int, int]


@dataclass(frozen=True)
class MoveResult:
    """The verified outcome of a requested movement."""

    moved: bool
    reason: str
    previous_position: Position
    current_position: Position


@dataclass(frozen=True)
class ScheduledObstacleEvent:
    """A deterministic obstacle change applied when the environment reaches a tick."""

    tick: int
    action: str
    position: Position
    destination: Position | None = None
    duration: int | None = None


class Environment:
    """A rectangular grid with static and runtime-added obstacles."""

    DIRECTIONS: dict[str, Position] = {
        "up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0),
    }
    DEFAULT_OBSTACLES: tuple[Position, ...] = (
        (2, 0), (2, 1), (2, 2), (4, 4), (4, 5), (4, 6),
        (6, 7), (7, 7), (8, 7),
    )

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        agent_position: Iterable[int] = (0, 0),
        goal_position: Iterable[int] = (9, 9),
        obstacles: Iterable[Iterable[int]] | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Grid width and height must be positive.")
        self.width = width
        self.height = height
        self.agent_position = self._as_position(agent_position)
        self.goal_position = self._as_position(goal_position)
        self._validate_position(self.agent_position)
        self._validate_position(self.goal_position)

        initial_obstacles = self.DEFAULT_OBSTACLES if obstacles is None else obstacles
        self._static_obstacles = {self._as_position(position) for position in initial_obstacles}
        for obstacle in self._static_obstacles:
            self._validate_position(obstacle)
        if self.agent_position in self._static_obstacles or self.goal_position in self._static_obstacles:
            raise ValueError("The agent and goal cannot occupy an obstacle.")

        self._dynamic_obstacles: dict[Position, int | None] = {}
        self._scheduled_events: list[ScheduledObstacleEvent] = []
        self._transient_move_failures = 0
        self.tick = 0
        self.state_version = 0
        self.events: list[str] = []

    @property
    def obstacles(self) -> list[list[int]]:
        """Current obstacles in the list format expected by the existing BFS code."""
        return [list(position) for position in sorted(self._all_obstacles())]

    def move(self, direction: str) -> MoveResult:
        """Attempt one movement and return a verified outcome without printing."""
        previous_position = self.agent_position
        if self._transient_move_failures:
            self._transient_move_failures -= 1
            self.events.append(f"move_failed:transient_failure:{previous_position}")
            return MoveResult(False, "transient_failure", previous_position, previous_position)
        if direction not in self.DIRECTIONS:
            return self._record_move(False, "invalid_direction", previous_position)
        dx, dy = self.DIRECTIONS[direction]
        target = (previous_position[0] + dx, previous_position[1] + dy)
        if not self.in_bounds(target):
            return self._record_move(False, "outside_grid", previous_position)
        if target in self._all_obstacles():
            return self._record_move(False, "obstacle", previous_position)

        self.agent_position = target
        self.state_version += 1
        self.events.append(f"move:{direction}:{previous_position}->{target}")
        return MoveResult(True, "moved", previous_position, target)

    def set_transient_move_failures(self, count: int) -> None:
        """Configure a deterministic number of injected move failures for experiments."""
        if count < 0:
            raise ValueError("Failure count cannot be negative.")
        self._transient_move_failures = count
        self.events.append(f"transient_move_failures_configured:{count}")

    def add_dynamic_obstacle(self, position: Iterable[int], duration: int | None = None) -> None:
        """Add a runtime obstacle, optionally expiring after ``duration`` updates."""
        obstacle = self._as_position(position)
        self._validate_position(obstacle)
        if obstacle == self.agent_position or obstacle == self.goal_position:
            raise ValueError("Cannot place a dynamic obstacle on the agent or goal.")
        if duration is not None and duration <= 0:
            raise ValueError("Obstacle duration must be positive or None.")
        self._dynamic_obstacles[obstacle] = duration
        self.state_version += 1
        self.events.append(f"dynamic_obstacle_added:{obstacle}:duration={duration}")

    def remove_dynamic_obstacle(self, position: Iterable[int]) -> bool:
        """Remove a runtime obstacle and report whether one was present."""
        obstacle = self._as_position(position)
        if obstacle not in self._dynamic_obstacles:
            return False
        del self._dynamic_obstacles[obstacle]
        self.state_version += 1
        self.events.append(f"dynamic_obstacle_removed:{obstacle}")
        return True

    def move_dynamic_obstacle(self, position: Iterable[int], destination: Iterable[int]) -> bool:
        """Move one existing dynamic obstacle to an unoccupied valid position."""
        source, target = self._as_position(position), self._as_position(destination)
        if source not in self._dynamic_obstacles:
            return False
        self._validate_position(target)
        if target in self._all_obstacles() or target in {self.agent_position, self.goal_position}:
            raise ValueError("Dynamic obstacle destination is occupied or reserved.")
        duration = self._dynamic_obstacles.pop(source)
        self._dynamic_obstacles[target] = duration
        self.state_version += 1
        self.events.append(f"dynamic_obstacle_moved:{source}->{target}")
        return True

    def schedule_obstacle_event(
        self, tick: int, action: str, position: Iterable[int],
        destination: Iterable[int] | None = None, duration: int | None = None,
    ) -> None:
        """Schedule an add, remove, or move event for reproducible simulations."""
        if tick <= self.tick:
            raise ValueError("Scheduled tick must be in the future.")
        if action not in {"add", "remove", "move"}:
            raise ValueError("Scheduled action must be 'add', 'remove', or 'move'.")
        source = self._as_position(position)
        target = self._as_position(destination) if destination is not None else None
        if action == "move" and target is None:
            raise ValueError("A move event requires a destination.")
        event = ScheduledObstacleEvent(tick, action, source, target, duration)
        self._scheduled_events.append(event)
        self._scheduled_events.sort(key=lambda item: item.tick)
        self.events.append(f"obstacle_event_scheduled:{event}")

    def advance(self) -> None:
        """Advance one environment update and expire temporary obstacles."""
        self.tick += 1
        expired: list[Position] = []
        for position, remaining in self._dynamic_obstacles.items():
            if remaining is not None:
                self._dynamic_obstacles[position] = remaining - 1
                if remaining == 1:
                    expired.append(position)
        for position in expired:
            del self._dynamic_obstacles[position]
            self.events.append(f"dynamic_obstacle_expired:{position}")
        due_events = [event for event in self._scheduled_events if event.tick == self.tick]
        self._scheduled_events = [event for event in self._scheduled_events if event.tick != self.tick]
        for event in due_events:
            if event.action == "add":
                self.add_dynamic_obstacle(event.position, event.duration)
            elif event.action == "remove":
                self.remove_dynamic_obstacle(event.position)
            else:
                self.move_dynamic_obstacle(event.position, event.destination)
            self.events.append(f"obstacle_event_applied:{event}")
        self.state_version += 1

    def get_state(self) -> dict[str, object]:
        """Return a copy-safe state snapshot for controllers and diagnostics."""
        return {
            "width": self.width, "height": self.height,
            "agent_position": self.agent_position, "goal_position": self.goal_position,
            "obstacles": tuple(sorted(self._all_obstacles())), "tick": self.tick,
            "state_version": self.state_version,
        }

    def obstacle_info(self, position: Iterable[int]) -> dict[str, object]:
        """Return non-secret obstacle metadata used for failure classification."""
        obstacle = self._as_position(position)
        if obstacle in self._dynamic_obstacles:
            remaining = self._dynamic_obstacles[obstacle]
            return {"present": True, "kind": "dynamic", "temporary": remaining is not None, "remaining_duration": remaining}
        return {"present": obstacle in self._static_obstacles, "kind": "static" if obstacle in self._static_obstacles else None, "temporary": False, "remaining_duration": None}

    def display(self, path: Iterable[Iterable[int]] | None = None) -> None:
        """Print the current grid, optionally overlaying a proposed route."""
        path_positions = {self._as_position(position) for position in path or ()}
        print("\nEnvironment Map:")
        print("A = Agent | G = Goal | # = Obstacle | * = Path | . = Empty")
        obstacles = self._all_obstacles()
        for y in range(self.height):
            row: list[str] = []
            for x in range(self.width):
                position = (x, y)
                if position == self.agent_position:
                    row.append(" A ")
                elif position == self.goal_position:
                    row.append(" G ")
                elif position in obstacles:
                    row.append(" # ")
                elif position in path_positions:
                    row.append(" * ")
                else:
                    row.append(" . ")
            print("".join(row))

    def in_bounds(self, position: Iterable[int]) -> bool:
        x, y = self._as_position(position)
        return 0 <= x < self.width and 0 <= y < self.height

    def _all_obstacles(self) -> set[Position]:
        return self._static_obstacles | set(self._dynamic_obstacles)

    def _record_move(self, moved: bool, reason: str, previous_position: Position) -> MoveResult:
        self.events.append(f"move_failed:{reason}:{previous_position}")
        return MoveResult(moved, reason, previous_position, previous_position)

    def _validate_position(self, position: Position) -> None:
        if not self.in_bounds(position):
            raise ValueError(f"Position {position} is outside the grid.")

    @staticmethod
    def _as_position(position: Iterable[int]) -> Position:
        values = tuple(position)
        if len(values) != 2 or not all(isinstance(value, int) for value in values):
            raise ValueError("Positions must contain exactly two integers.")
        return values[0], values[1]
