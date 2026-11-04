"""Three-strategy, byte-reproducible Milestone 11 verification evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean

from agent import AutonomousAgent
from evaluation import EvaluationRunner
from milestone11_scenarios import validation_factories
from outcome_verifier import VerificationState, progress_verified
from recovery import RecoveryAction


@dataclass(frozen=True)
class OutcomeTrial:
    strategy: str; scenario: str; trial: int; reached_goal: bool; action_executed: int; environment_changed: int
    progress_restored: int; task_completed: int; recovery_failed: int; false_positive_recovery_success: int
    safe_stops: int; failed_actions: int; recovery_attempts: int; retries: int; replans: int; human_requests: int; recovery_ticks: int


class Milestone11Evaluator:
    def __init__(self, trials_per_scenario: int = 10) -> None:
        self.trials_per_scenario = trials_per_scenario
        self.factories = validation_factories()
        self._immediate = EvaluationRunner()

    def environments_are_equivalent(self) -> bool:
        for factory in self.factories.values():
            first, second, third = factory(), factory(), factory()
            if not (first.get_state() == second.get_state() == third.get_state() and first.events == second.events == third.events):
                return False
        return True

    def run(self) -> list[OutcomeTrial]:
        results = []
        for name, factory in self.factories.items():
            for trial in range(1, self.trials_per_scenario + 1):
                immediate = self._immediate._run_immediate(name, trial, factory())
                results.append(OutcomeTrial("Immediate replanning", name, trial, immediate.reached_goal, 0, 0, 0, int(immediate.reached_goal), 0, 0, int(not immediate.reached_goal), immediate.failed_actions, immediate.recovery_attempts, 0, immediate.replans, 0, 0))
                results.append(self._run_agent("Historical Milestone 9", name, trial, factory(), "m9"))
                results.append(self._run_agent("Milestone 11 candidate", name, trial, factory(), "m11"))
        return results

    def _run_agent(self, strategy, scenario, trial, environment, mode):
        outcome = AutonomousAgent(environment, logger=lambda _: None, recovery_mode=mode).run()
        records = outcome.recoveries
        count = lambda state: sum(state in record.verification_states for record in records)
        false_positive = sum(
            record.succeeded
            and not progress_verified(record.verification_states)
            and VerificationState.RECOVERY_FAILED not in record.verification_states
            for record in records
        )
        return OutcomeTrial(strategy, scenario, trial, outcome.reached_goal,
            count(VerificationState.ACTION_EXECUTED), count(VerificationState.ENVIRONMENT_CHANGED),
            count(VerificationState.PROGRESS_RESTORED), int(outcome.reached_goal),
            count(VerificationState.RECOVERY_FAILED), false_positive, int(not outcome.reached_goal),
            outcome.failed_moves, len(records), sum(record.decision.action == RecoveryAction.RETRY for record in records),
            sum(record.decision.action == RecoveryAction.REPLAN for record in records),
            sum(record.decision.action == RecoveryAction.REQUEST_HUMAN_HELP for record in records), sum(record.ticks_spent for record in records))

    @staticmethod
    def byte_equivalent(first: list[OutcomeTrial], second: list[OutcomeTrial]) -> bool:
        return repr([asdict(item) for item in first]).encode() == repr([asdict(item) for item in second]).encode()

    def summary(self, results):
        report = {}
        for strategy in ("Immediate replanning", "Historical Milestone 9", "Milestone 11 candidate"):
            group = [item for item in results if item.strategy == strategy]
            report[strategy] = {
                "completion_rate": mean(item.reached_goal for item in group), "recovery_success_rate": mean(item.progress_restored > 0 for item in group),
                "action_executed": sum(item.action_executed for item in group), "environment_changed": sum(item.environment_changed for item in group),
                "progress_restored": sum(item.progress_restored for item in group), "task_completed": sum(item.task_completed for item in group),
                "recovery_failed": sum(item.recovery_failed for item in group), "false_positive": sum(item.false_positive_recovery_success for item in group),
                "safe_stops": sum(item.safe_stops for item in group), "failed_actions": sum(item.failed_actions for item in group),
                "recovery_attempts": sum(item.recovery_attempts for item in group), "retries": sum(item.retries for item in group),
                "replans": sum(item.replans for item in group), "human_requests": sum(item.human_requests for item in group),
                "average_recovery_ticks": sum(item.recovery_ticks for item in group) / sum(item.recovery_attempts for item in group) if sum(item.recovery_attempts for item in group) else 0.0,
            }
        return report


def main() -> None:
    """Run the complete deterministic evaluation and print its existing metrics."""
    evaluator = Milestone11Evaluator(trials_per_scenario=10)
    equivalent_environments = evaluator.environments_are_equivalent()
    first_run = evaluator.run()
    second_run = evaluator.run()
    reproducible = evaluator.byte_equivalent(first_run, second_run)
    summary = evaluator.summary(first_run)

    print("REROUTE Milestone 11 Evaluation")
    print(f"Total trial count: {len(first_run)}")
    print(f"Equivalent initial environments: {equivalent_environments}")
    for strategy, metrics in summary.items():
        print(f"\n{strategy}")
        print(f"  Task completion rate: {metrics['completion_rate']:.1%}")
        print(f"  Recovery success rate: {metrics['recovery_success_rate']:.1%}")
        print(f"  False-positive recoveries: {metrics['false_positive']}")
        print(f"  Failed actions: {metrics['failed_actions']}")
        print(f"  Retries: {metrics['retries']}")
        print(f"  Replans: {metrics['replans']}")
        print(f"  Human requests: {metrics['human_requests']}")
    print(f"\nByte-equivalent reproducibility: {reproducible}")


if __name__ == "__main__":
    main()
