from controller import Robot

# Initialize Webots robot
robot = Robot()

# Simulation timestep
timestep = int(robot.getBasicTimeStep())

# Initialize motors
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))

# Keep the robot stationary during the test
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Initialize proximity sensors
sensors = []

for i in range(8):
    sensor = robot.getDevice(f"ps{i}")
    sensor.enable(timestep)
    sensors.append(sensor)

print("================================")
print("REROUTE M14: PYTHON CONTROLLER")
print("E-puck initialized successfully")
print("8 proximity sensors enabled")
print("Motors initialized and stopped")
print("================================")

# Main simulation loop
while robot.step(timestep) != -1:

    sensor_values = [
        round(sensor.getValue(), 2)
        for sensor in sensors
    ]

    print("Proximity:", sensor_values)