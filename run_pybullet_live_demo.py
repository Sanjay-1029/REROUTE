import os
import json
import time
import argparse
import traceback
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from pybullet_environment import PyBulletEnvironment
from agent import AutonomousAgent
from simulation_controller import reroute_decision_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--image", default="test_image.jpg")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()

    if args.live:
        if not os.getenv("MODEL_API_KEY"):
            raise RuntimeError("MODEL_API_KEY is missing. Live mode cannot start.")
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        os.environ["REROUTE_IMAGE_PATH"] = str(image_path)

    live_request_used = False
    decision_log = []
    env = None

    report = {
        "project": "REROUTE",
        "demonstration": "PyBullet + Muse Spark",
        "mode": "LIVE" if args.live else "OFFLINE",
        "model": "muse-spark-1.3" if args.live else "OFFLINE MOCK",
        "maximum_live_requests": 1,
        "decision_pipeline_results": decision_log,
    }

    def decision_policy(context):
        nonlocal live_request_used

        use_live = args.live and not live_request_used

        if use_live:
            live_request_used = True
            print("\n[LIVE] Sending one recovery request to Muse Spark.")
        else:
            print("\n[OFFLINE] Using local mock decision pipeline.")

        try:
            result = reroute_decision_policy(
                context,
                live_reasoning=use_live
            )

            # Preserve the original recommendation and final evaluated decision.
            decision_log.append(result)

            print("\n[DECISION EVIDENCE]")
            print("Pipeline status:", result.get("status"))
            print(
                "Model recommendation:",
                result.get("original_recommendation", {}).get(
                    "recommended_action", "Not available"
                )
            )
            print("REROUTE selected:", result.get("selected_action", "Not available"))

            return result

        except Exception as error:
            decision_log.append({
                "status": "ERROR",
                "error": str(error),
                "live_request": use_live,
            })
            raise

    try:
        print("\n========== REROUTE LIVE DEMONSTRATION ==========")

        if args.live:
            print("Muse Spark: ENABLED")
            print("Image:", image_path.name)
        else:
            print("Muse Spark: DISABLED")

        env = PyBulletEnvironment(
            width=10,
            height=10,
            agent_position=(0, 0),
            goal_position=(9, 9),
            gui=True,
            realtime=True,
        )

        print("\nInjecting one transient movement failure...")
        env.set_transient_move_failures(1)

        agent = AutonomousAgent(
            environment=env,
            algorithm="astar",
            recovery_mode="m11",
            decision_policy=decision_policy,
        )

        start_time = time.time()
        outcome = agent.run()
        elapsed = round(time.time() - start_time, 2)

        report.update({
            "status": "PASSED" if outcome.reached_goal else "INCOMPLETE",
            "goal_reached": outcome.reached_goal,
            "reason": outcome.reason,
            "final_position": list(env.agent_position),
            "successful_moves": outcome.successful_moves,
            "failed_moves": outcome.failed_moves,
            "recovery_events": len(outcome.recoveries),
            "live_request_attempted": live_request_used,
            "elapsed_seconds": elapsed,
            "recoveries": [str(r) for r in outcome.recoveries],
        })

        print("\n========== FINAL RESULTS ==========")
        print("Goal reached:", outcome.reached_goal)
        print("Final position:", env.agent_position)
        print("Successful moves:", outcome.successful_moves)
        print("Failed moves:", outcome.failed_moves)
        print("Recovery events:", len(outcome.recoveries))
        print("Live request attempted:", live_request_used)
        print("Elapsed:", elapsed, "seconds")

    except Exception as error:
        report["status"] = "ERROR"
        report["error"] = str(error)
        traceback.print_exc()

    finally:
        Path("reroute_live_pybullet_report.json").write_text(
            json.dumps(report, indent=4, default=str),
            encoding="utf-8",
        )

        print("\nReport saved: reroute_live_pybullet_report.json")

        if env is not None:
            input("\nPress ENTER to close the simulator...")
            env.close()


if __name__ == "__main__":
    main()
