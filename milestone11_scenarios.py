"""Separate Milestone 11 validation configurations; prior benchmarks are untouched."""

from environment import Environment


def validation_factories():
    factories = {f"temporary_{duration}": _temporary(duration) for duration in (1, 2, 3, 5, 8)}
    factories.update({
        "multiple_temporary": _multiple_temporary,
        "moving_during_wait": _moving_during_wait,
        "transient_1": _transient(1), "transient_2": _transient(2),
        "transient_3": _transient(3), "transient_5": _transient(5),
        "wait_without_progress": _wait_without_progress,
        "route_invalidated_before_move": _route_invalidated,
        "recovery_then_goal": _temporary(1),
        "recovery_limit_exhausted": _transient(5),
    })
    return factories


def _temporary(duration):
    def factory():
        environment = Environment(1, 3, (0, 0), (0, 2), [])
        environment.schedule_obstacle_event(1, "add", (0, 1), duration=duration)
        return environment
    return factory


def _multiple_temporary():
    environment = Environment(3, 3, (0, 0), (2, 0), [(0, 1), (1, 1), (2, 1)])
    environment.schedule_obstacle_event(1, "add", (1, 0), duration=2)
    return environment


def _moving_during_wait():
    environment = Environment(3, 3, (0, 0), (2, 0), [(0, 1), (2, 1)])
    environment.schedule_obstacle_event(1, "add", (1, 0), duration=3)
    environment.schedule_obstacle_event(2, "move", (1, 0), destination=(1, 1))
    return environment


def _transient(count):
    def factory():
        environment = Environment()
        environment.set_transient_move_failures(count)
        return environment
    return factory


def _wait_without_progress():
    environment = Environment(1, 4, (0, 0), (0, 3), [])
    environment.schedule_obstacle_event(1, "add", (0, 1), duration=1)
    environment.schedule_obstacle_event(2, "add", (0, 2))
    return environment


def _route_invalidated():
    environment = Environment()
    environment.schedule_obstacle_event(1, "add", (0, 1))
    return environment
