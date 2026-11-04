import inspect
from agent import AutonomousAgent
from environment import Environment

methods = [
    AutonomousAgent.__init__,
    AutonomousAgent.plan_route,
    AutonomousAgent.run,
    AutonomousAgent.execute_next,
    AutonomousAgent._recover_m11,
    Environment.move,
    Environment.get_state,
]

with open("REROUTE_AGENT_INTERFACE.txt", "w", encoding="utf-8") as f:
    for method in methods:
        f.write("\n" + "=" * 65 + "\n")
        f.write(method.__qualname__ + "\n")
        f.write("=" * 65 + "\n")
        f.write(inspect.getsource(method))

print("Inspection completed successfully.")
