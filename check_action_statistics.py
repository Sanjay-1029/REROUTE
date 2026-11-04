from collections import defaultdict
from evaluation import EvaluationRunner

runner = EvaluationRunner()
results = runner.run()

action_stats = defaultdict(lambda: {"success": 0, "failure": 0})

for trial in results:
    for action, succeeded in trial.recovery_outcomes:
        if succeeded:
            action_stats[action]["success"] += 1
        else:
            action_stats[action]["failure"] += 1

print("\nACTION SUCCESS STATISTICS")
print("-" * 40)

for action, stats in sorted(action_stats.items()):
    total = stats["success"] + stats["failure"]
    rate = (stats["success"] / total) * 100

    print(f"\nAction: {action}")
    print(f"Successes: {stats['success']}")
    print(f"Failures: {stats['failure']}")
    print(f"Total: {total}")
    print(f"Success rate: {rate:.2f}%")