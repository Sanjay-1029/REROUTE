"""Deterministic Milestone 7 policy and execution checks."""

from __future__ import annotations

from agent import AutonomousAgent
from diagnosis import DiagnosisCause, DiagnosisRecord
from environment import Environment
from failure_detector import FailureRecord, FailureType, Severity
from recovery import RecoveryAction, RecoveryEngine


def run_recovery_scenarios() -> dict[str, RecoveryAction]:
    """Exercise all five recovery choices, including bounded repeated attempts."""
    results: dict[str, RecoveryAction] = {}

    temporary = Environment()
    temporary.schedule_obstacle_event(1, "add", (0, 1), duration=1)
    wait_run = AutonomousAgent(temporary, logger=lambda _: None).run()
    assert wait_run.reached_goal and wait_run.recoveries[0].succeeded
    results["wait"] = wait_run.recoveries[0].decision.action

    permanent = Environment()
    permanent.schedule_obstacle_event(1, "add", (0, 1))
    replan_run = AutonomousAgent(permanent, logger=lambda _: None).run()
    assert replan_run.reached_goal and replan_run.recoveries[0].succeeded
    results["replan"] = replan_run.recoveries[0].decision.action

    environment = Environment(width=3, height=3, agent_position=(0, 0), goal_position=(2, 2), obstacles=[])
    retry_diagnosis = _diagnosis(environment, DiagnosisCause.TEMPORARY_OBSTACLE, 0.60)
    engine = RecoveryEngine()
    retry = engine.decide(retry_diagnosis, environment, True)
    movement = environment.move("right")
    engine.record(retry_diagnosis, retry, movement.reason, movement.moved)
    assert retry.action == RecoveryAction.RETRY and movement.moved
    results["retry"] = retry.action

    gather_diagnosis = _diagnosis(Environment(), DiagnosisCause.STALE_ENVIRONMENT_INFORMATION, 0.66)
    gather = RecoveryEngine().decide(gather_diagnosis, Environment(), True)
    assert gather.action == RecoveryAction.GATHER_INFORMATION
    results["gather_information"] = gather.action

    uncertain = _diagnosis(Environment(), DiagnosisCause.UNKNOWN_CAUSE, 0.28, human=True)
    human = RecoveryEngine().decide(uncertain, Environment(), False)
    assert human.action == RecoveryAction.REQUEST_HUMAN_HELP
    results["human_help"] = human.action

    limited = RecoveryEngine(max_attempts=2)
    repeat = _diagnosis(Environment(), DiagnosisCause.UNKNOWN_CAUSE, 0.50)
    first = limited.decide(repeat, Environment(), False)
    limited.record(repeat, first, "no_new_evidence", False)
    second = limited.decide(repeat, Environment(), False)
    limited.record(repeat, second, "no_new_evidence", False)
    assert limited.decide(repeat, Environment(), False).action == RecoveryAction.REQUEST_HUMAN_HELP
    return results


def _diagnosis(environment: Environment, cause: DiagnosisCause, confidence: float, human: bool = False) -> DiagnosisRecord:
    failure = FailureRecord(FailureType.MOVEMENT_BLOCKED, 0, environment.agent_position, (1, 0), environment.get_state(), "Controlled recovery scenario.", Severity.WARNING, cause == DiagnosisCause.TEMPORARY_OBSTACLE)
    return DiagnosisRecord(failure, cause, confidence, ("Controlled evidence.",), ("Additional observations may help.",), "Inspect state.", human)
