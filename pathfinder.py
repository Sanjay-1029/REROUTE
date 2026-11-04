"""Deterministic grid pathfinding algorithms for REROUTE."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from typing import Iterable

Position = tuple[int, int]
DIRECTIONS: tuple[Position, ...] = ((0, -1), (0, 1), (-1, 0), (1, 0))


@dataclass(frozen=True)
class SearchResult:
    algorithm: str
    path: list[list[int]] | None
    explored_nodes: int

    @property
    def steps(self) -> int | None:
        return len(self.path) - 1 if self.path else None


def bfs_search(start: Iterable[int], goal: Iterable[int], obstacles: Iterable[Iterable[int]], width: int, height: int) -> SearchResult:
    """Find a shortest unweighted route with breadth-first search."""
    start, goal, blocked = _prepare(start, goal, obstacles, width, height)
    queue = deque([(start, [start])])
    visited = {start}
    explored = 0
    while queue:
        current, path = queue.popleft()
        explored += 1
        if current == goal:
            return SearchResult("BFS", _to_lists(path), explored)
        for neighbor in _neighbors(current, blocked, width, height):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return SearchResult("BFS", None, explored)


def astar_search(start: Iterable[int], goal: Iterable[int], obstacles: Iterable[Iterable[int]], width: int, height: int) -> SearchResult:
    """Find a shortest route using Manhattan-distance A* on this 4-way grid."""
    start, goal, blocked = _prepare(start, goal, obstacles, width, height)
    sequence = count()
    frontier: list[tuple[int, int, Position]] = [(0, next(sequence), start)]
    came_from: dict[Position, Position | None] = {start: None}
    cost = {start: 0}
    explored = 0
    while frontier:
        _, _, current = heappop(frontier)
        explored += 1
        if current == goal:
            path = _reconstruct(came_from, goal)
            return SearchResult("A*", _to_lists(path), explored)
        for neighbor in _neighbors(current, blocked, width, height):
            new_cost = cost[current] + 1
            if new_cost < cost.get(neighbor, float("inf")):
                cost[neighbor] = new_cost
                came_from[neighbor] = current
                priority = new_cost + _manhattan(neighbor, goal)
                heappush(frontier, (priority, next(sequence), neighbor))
    return SearchResult("A*", None, explored)


def find_path(start, goal, obstacles, width, height):
    """Legacy BFS API returning only the path."""
    return bfs_search(start, goal, obstacles, width, height).path


def _prepare(start: Iterable[int], goal: Iterable[int], obstacles: Iterable[Iterable[int]], width: int, height: int) -> tuple[Position, Position, set[Position]]:
    if width <= 0 or height <= 0:
        raise ValueError("Grid width and height must be positive.")
    start_position, goal_position = _position(start), _position(goal)
    blocked = {_position(obstacle) for obstacle in obstacles}
    if not _in_bounds(start_position, width, height) or not _in_bounds(goal_position, width, height):
        raise ValueError("Start and goal must be inside the grid.")
    return start_position, goal_position, blocked


def _neighbors(position: Position, blocked: set[Position], width: int, height: int):
    for dx, dy in DIRECTIONS:
        neighbor = (position[0] + dx, position[1] + dy)
        if _in_bounds(neighbor, width, height) and neighbor not in blocked:
            yield neighbor


def _reconstruct(came_from: dict[Position, Position | None], goal: Position) -> list[Position]:
    path = [goal]
    while came_from[path[-1]] is not None:
        path.append(came_from[path[-1]])
    return list(reversed(path))


def _position(value: Iterable[int]) -> Position:
    x, y = tuple(value)
    return x, y


def _in_bounds(position: Position, width: int, height: int) -> bool:
    return 0 <= position[0] < width and 0 <= position[1] < height


def _to_lists(path: list[Position]) -> list[list[int]]:
    return [list(position) for position in path]


def _manhattan(first: Position, second: Position) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])
