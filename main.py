"""Run REROUTE scenarios. Offline by default; --live makes at most ONE Meta call."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from scenarios import run_dynamic_scenarios
from simulation_controller import reroute_decision_policy


def main():
    parser = argparse.ArgumentParser(description="Run REROUTE autonomous scenarios.")
    parser.add_argument("--live", action="store_true", help="Use Muse Spark for at most one recovery decision (one API request).")
    parser.add_argument("--image", type=Path, help="Optional local image attached to that one live request.")
    args = parser.parse_args()

    print("=" * 48)
    print("          REROUTE AUTONOMOUS AGENT")
    print("=" * 48)
    if args.image:
        image_path = args.image.expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        os.environ["REROUTE_IMAGE_PATH"] = str(image_path)
        print(f"Image configured for Muse Spark: {image_path.name}")

    if args.live:
        print("REASONING MODE: LIVE MUSE SPARK, hard limit = ONE API request for this run.")
    else:
        print("REASONING MODE: OFFLINE MOCK. No Meta API requests will be made.")

    live_call_used = False
    def policy(context):
        nonlocal live_call_used
        use_live = args.live and not live_call_used
        if use_live:
            live_call_used = True
        return reroute_decision_policy(context, live_reasoning=use_live)

    results = run_dynamic_scenarios(logger=print, decision_policy=policy)
    print("\n========== FINAL RESULTS ==========")
    for result in results:
        print(f"\nScenario: {result.name}")
        print(f"Initial route: {result.initial_route}")
        print(f"Result: {result.outcome.reason}")
        print(f"Goal reached: {result.outcome.reached_goal}")
        print(f"Invalidated positions: {result.outcome.invalidated_positions}")
        if "Temporary" in result.name:
            print(f"Temporary obstacle cleared: {result.temporary_obstacle_cleared}")


if __name__ == "__main__":
    main()
