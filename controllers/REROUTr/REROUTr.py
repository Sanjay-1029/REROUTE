from controller import Robot

# ==========================================
# REROUTE - Autonomous Robot Controller
# Milestone 14.1 - Webots Integration
# ==========================================

robot = Robot()

TIME_STEP = int(robot.getBasicTimeStep())

# Robot movement configuration
MAX_SPEED = 6.28
FORWARD_SPEED = 2.0
TURN_SPEED = 2.0

# ==========================================
# 1. Initialize Motors
# ==========================================

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# ==========================================
# 2. Initialize Proximity Sensors
# ==========================================

sensors = []

for i in range(8):
    sensor = robot.getDevice(f"ps{i}")
    sensor.enable(TIME_STEP)
    sensors.append(sensor)

print("\n================================")
print("       REROUTE INITIALIZED")
print("================================")
print("Robot: E-puck")
print("Motors: Connected")
print("Proximity sensors: 8")
print("Autonomous navigation: Active")
print("================================\n")

# ==========================================
# 3. Main Autonomous Loop
# ==========================================

while robot.step(TIME_STEP) != -1:

    # Read sensor values
    readings = [sensor.getValue() for sensor in sensors]

    # Front-facing proximity sensors
    front_left = readings[0] + readings[1]
    front_right = readings[6] + readings[7]

    front_total = front_left + front_right

    # Obstacle detection threshold
    OBSTACLE_THRESHOLD = 100

    # ======================================
    # 4. Decision Making
    # ======================================

    if front_total > OBSTACLE_THRESHOLD:

        print("\n[REROUTE] Obstacle detected!")
        print("[REROUTE] Evaluating alternative direction...")

        # Stop before turning
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

        # Turn away from the stronger sensor response
        if front_left > front_right:

            print("[REROUTE] Turning right")

            left_motor.setVelocity(TURN_SPEED)
            right_motor.setVelocity(-TURN_SPEED)

        else:

            print("[REROUTE] Turning left")

            left_motor.setVelocity(-TURN_SPEED)
            right_motor.setVelocity(TURN_SPEED)

    else:

        # Clear path: move forward
        left_motor.setVelocity(FORWARD_SPEED)
        right_motor.setVelocity(FORWARD_SPEED)

    # ======================================
    # 5. Monitoring
    # ======================================

    print(
        "Sensors:",
        [round(value, 1) for value in readings],
        "| Front:",
        round(front_total, 1)
    )

# ==========================================
# 6. Safe Shutdown
# ==========================================

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

print("REROUTE stopped safely.")