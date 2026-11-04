from pybullet_environment import PyBulletEnvironment
from agent import AutonomousAgent
from simulation_controller import reroute_decision_policy

env = PyBulletEnvironment(gui=True)

try:
    env.set_transient_move_failures(1)

    agent = AutonomousAgent(
        environment=env,
        algorithm="astar",
        recovery_mode="m11",
        decision_policy=reroute_decision_policy,
    )

    result = agent.run()

    print("\n========== M14 RECOVERY VALIDATION ==========")
    print("Goal reached:", result.reached_goal)
    print("Reason:", result.reason)
    print("Final position:", env.agent_position)
    print("Successful moves:", result.successful_moves)
    print("Failed moves:", result.failed_moves)
    print("Recovery records:", len(result.recoveries))

    for recovery in result.recoveries:
        print(recovery)

    print("=============================================")

    input("\nPress ENTER to close simulator...")

finally:
    env.close()
