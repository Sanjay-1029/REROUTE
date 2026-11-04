import time
from pybullet_environment import PyBulletEnvironment

env = PyBulletEnvironment(gui=True)

try:
    print("\nREROUTE SIMULATOR RUNNING")
    print("Starting position:", env.agent_position)

    for direction in ["down", "down", "right", "right"]:
        result = env.move(direction)
        print(result)
        print("Sensors:", env.read_proximity_sensors())
        time.sleep(0.5)

    print("\nSimulation will remain open for inspection.")
    input("Press ENTER to close the simulator...")

finally:
    env.close()
