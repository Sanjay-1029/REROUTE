"""Run the existing REROUTE autonomous agent inside PyBullet."""

from pathlib import Path
import json
import time
import traceback

from pybullet_environment import PyBulletEnvironment
from agent import AutonomousAgent
from simulation_controller import reroute_decision_policy


def main():
    env = None

    report = {
        "project": "REROUTE",
        "milestone": "M14 - PyBullet Simulator Integration",
        "simulator": "PyBullet",
        "agent": "AutonomousAgent",
        "planner": "A*",
        "recovery_mode": "M11",
        "decision_provider": "Offline REROUTE decision pipeline",
    }

    try:
        print("\n" + "=" * 60)
        print("       REROUTE AUTONOMOUS SIMULATOR")
        print("=" * 60)

        env = PyBulletEnvironment(
            width=10,
            height=10,
            agent_position=(0, 0),
            goal_position=(9, 9),
            gui=True,
            realtime=True,
        )

        print("\n[1/5] Physical environment initialized.")

        initial_sensors = env.read_proximity_sensors()
        print("\n[2/5] Initial proximity readings:")
        for sensor, distance in initial_sensors.items():
            print(f"  {sensor:15}: {distance:.3f} m")

        print("\n[3/5] Connecting REROUTE autonomous agent...")

        agent = AutonomousAgent(
            environment=env,
            algorithm="astar",
            recovery_mode="m11",
            decision_policy=reroute_decision_policy,
        )

        print("Agent connected successfully.")
        print("\n[4/5] Starting autonomous navigation...")
        print("Start:", env.agent_position)
        print("Goal:", env.goal_position)
        print("Obstacles:", len(env.obstacles))
        print("\n" + "-" * 60)

        start_time = time.time()
        outcome = agent.run()
        elapsed = round(time.time() - start_time, 2)

        print("\n" + "-" * 60)
        print("[5/5] AUTONOMOUS RUN COMPLETED")
        print("-" * 60)

        print("Goal reached:", outcome.reached_goal)
        print("Reason:", outcome.reason)
        print("Final position:", env.agent_position)
        print("Successful moves:", outcome.successful_moves)
        print("Failed moves:", outcome.failed_moves)
        print("Recovery events:", len(outcome.recoveries))
        print("Elapsed seconds:", elapsed)

        print("\nMovement history:")
        for action in outcome.actions:
            print(
                f"  Step {action.step}: "
                f"{action.direction} | "
                f"{action.actual_position} | "
                f"{'SUCCESS' if action.succeeded else 'FAILED'}"
            )

        final_sensors = env.read_proximity_sensors()

        report.update({
            "goal_reached": outcome.reached_goal,
            "result_reason": outcome.reason,
            "final_position": list(env.agent_position),
            "successful_moves": outcome.successful_moves,
            "failed_moves": outcome.failed_moves,
            "recovery_events": len(outcome.recoveries),
            "elapsed_seconds": elapsed,
            "initial_sensors": initial_sensors,
            "final_sensors": final_sensors,
            "movement_history": [
                {
                    "step": a.step,
                    "direction": a.direction,
                    "position": list(a.actual_position),
                    "success": a.succeeded,
                    "reason": a.reason,
                }
                for a in outcome.actions
            ],
        })

        report["status"] = (
            "PASSED" if outcome.reached_goal else "INCOMPLETE"
        )

    except Exception as error:
        report["status"] = "ERROR"
        report["error"] = str(error)
        traceback.print_exc()

    finally:
        Path("m14_pybullet_report.json").write_text(
            json.dumps(report, indent=4),
            encoding="utf-8",
        )

        print("\nReport saved: m14_pybullet_report.json")

        if env is not None:
            try:
                input("\nPress ENTER to close the PyBullet simulator...")
            finally:
                env.close()

        print("\nM14 simulator run finished.")


if __name__ == "__main__":
    main()
