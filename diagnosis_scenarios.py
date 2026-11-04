"""Deterministic evidence cases for all Milestone 6 diagnosis causes."""

from __future__ import annotations

from agent import AutonomousAgent
from diagnosis import DiagnosisCause, FailureDiagnosis
from environment import Environment, MoveResult
from failure_detector import FailureDetector, FailureRecord, FailureType, Severity


def run_diagnosis_scenarios() -> dict[str, DiagnosisCause]:
    diagnoses: dict[str, DiagnosisCause] = {}
    engine = FailureDiagnosis()

    temporary = Environment(width=3, height=3, agent_position=(0, 0), goal_position=(2, 2), obstacles=[])
    temporary.add_dynamic_obstacle((1, 0), duration=2)
    failure = FailureDetector().observe_movement(temporary, temporary.move("right"), (1, 0))
    diagnoses["temporary"] = engine.diagnose(failure, temporary).cause

    persistent = Environment(width=3, height=3, agent_position=(0, 0), goal_position=(2, 2), obstacles=[(1, 0)])
    failure = FailureDetector().observe_movement(persistent, persistent.move("right"), (1, 0))
    diagnoses["persistent"] = engine.diagnose(failure, persistent).cause

    invalid = Environment(width=3, height=3, agent_position=(0, 0), goal_position=(2, 2), obstacles=[])
    failure = FailureDetector().observe_movement(invalid, invalid.move("bad_direction"), (1, 0))
    diagnoses["invalid"] = engine.diagnose(failure, invalid).cause

    stale = Environment(width=3, height=3, agent_position=(0, 0), goal_position=(2, 2), obstacles=[])
    old = FailureRecord(FailureType.ROUTE_OUTDATED, 0, (0, 0), (1, 0), stale.get_state(), "Planned route now conflicts with observation.", Severity.ERROR, False)
    stale.add_dynamic_obstacle((1, 0))
    diagnoses["stale"] = engine.diagnose(old, stale).cause

    trapped = Environment(width=3, height=3, agent_position=(0, 0), goal_position=(2, 2), obstacles=[(1, 0), (0, 1)])
    failure = AutonomousAgent(trapped, logger=lambda _: None).run().failures[0]
    diagnoses["no_route"] = engine.diagnose(failure, trapped).cause

    repeated = Environment(width=3, height=3, agent_position=(0, 0), goal_position=(2, 2), obstacles=[(1, 0)])
    detector = FailureDetector()
    for _ in range(2): detector.observe_movement(repeated, repeated.move("right"), (1, 0))
    diagnoses["repeated"] = engine.diagnose(detector.history[-1], repeated, previous_failures=detector.history).cause

    insufficient = FailureRecord(FailureType.MOVEMENT_BLOCKED, 0, (0, 0), None, {}, "Movement outcome was not captured.", Severity.WARNING, False)
    diagnoses["insufficient"] = engine.diagnose(insufficient).cause

    unknown = Environment(width=3, height=3, agent_position=(0, 0), goal_position=(2, 2), obstacles=[])
    unknown.agent_position = (1, 0)
    failure = FailureDetector().observe_movement(unknown, MoveResult(True, "moved", (0, 0), (1, 0)), (0, 1))
    diagnoses["unknown"] = engine.diagnose(failure, unknown).cause
    return diagnoses
