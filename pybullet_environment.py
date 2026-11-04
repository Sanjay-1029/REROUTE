"""PyBullet physical simulation adapter for the REROUTE agent."""

from __future__ import annotations

import math
import time
from typing import Iterable

import pybullet as p
import pybullet_data

from environment import Environment, MoveResult, Position


class PyBulletEnvironment(Environment):
    """Connects REROUTE's grid environment to a physical PyBullet world."""

    CELL_SIZE = 1.5
    ROBOT_RADIUS = 0.22
    ROBOT_HEIGHT = 0.35
    ROBOT_SPEED = 1.8
    SENSOR_RANGE = 2.0

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        agent_position: Iterable[int] = (0, 0),
        goal_position: Iterable[int] = (9, 9),
        obstacles=None,
        gui: bool = True,
        realtime: bool = True,
    ):
        super().__init__(
            width=width,
            height=height,
            agent_position=agent_position,
            goal_position=goal_position,
            obstacles=obstacles,
        )

        self.gui = gui
        self.realtime = realtime
        self.client = p.connect(p.GUI if gui else p.DIRECT)

        if self.client < 0:
            raise RuntimeError("Could not connect to PyBullet.")

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client)

        p.loadURDF("plane.urdf", physicsClientId=self.client)

        self.obstacle_bodies: dict[Position, int] = {}
        self._create_robot()
        self._create_goal_marker()
        self._sync_obstacles()

        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=max(width, height) * 1.5,
                cameraYaw=45,
                cameraPitch=-65,
                cameraTargetPosition=[
                    width * self.CELL_SIZE / 2,
                    -height * self.CELL_SIZE / 2,
                    0,
                ],
                physicsClientId=self.client,
            )

        print("REROUTE: PyBullet environment initialized.")
        print(f"Robot starting position: {self.agent_position}")
        print(f"Goal position: {self.goal_position}")
        print(f"Physical obstacles: {len(self.obstacle_bodies)}")

    def _world_position(self, position: Position) -> tuple[float, float]:
        x, y = position
        return x * self.CELL_SIZE, -y * self.CELL_SIZE

    def _create_robot(self):
        collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=self.ROBOT_RADIUS,
            height=self.ROBOT_HEIGHT,
            physicsClientId=self.client,
        )

        visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.ROBOT_RADIUS,
            length=self.ROBOT_HEIGHT,
            rgbaColor=[0.1, 0.35, 0.95, 1],
            physicsClientId=self.client,
        )

        x, y = self._world_position(self.agent_position)

        self.robot_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[x, y, self.ROBOT_HEIGHT / 2 + 0.02],
            physicsClientId=self.client,
        )

        p.changeDynamics(
            self.robot_id,
            -1,
            lateralFriction=1.0,
            linearDamping=0.05,
            angularDamping=0.9,
            physicsClientId=self.client,
        )

    def _create_goal_marker(self):
        x, y = self._world_position(self.goal_position)

        self.goal_id = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.35,
            length=0.035,
            rgbaColor=[0.1, 0.9, 0.2, 0.8],
            physicsClientId=self.client,
        )

        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self.goal_id,
            basePosition=[x, y, 0.025],
            physicsClientId=self.client,
        )

    def _create_obstacle(self, position: Position) -> int:
        x, y = self._world_position(position)

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.43, 0.43, 0.35],
            physicsClientId=self.client,
        )

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.43, 0.43, 0.35],
            rgbaColor=[0.85, 0.2, 0.15, 1],
            physicsClientId=self.client,
        )

        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[x, y, 0.35],
            physicsClientId=self.client,
        )

    def _sync_obstacles(self):
        current = self._all_obstacles()

        for position in list(self.obstacle_bodies):
            if position not in current:
                p.removeBody(
                    self.obstacle_bodies.pop(position),
                    physicsClientId=self.client,
                )

        for position in current:
            if position not in self.obstacle_bodies:
                self.obstacle_bodies[position] = self._create_obstacle(position)

    def read_proximity_sensors(self) -> dict[str, float]:
        """Return eight simulated ray-based proximity readings."""

        position, _ = p.getBasePositionAndOrientation(
            self.robot_id,
            physicsClientId=self.client,
        )

        directions = {
            "front": (1, 0),
            "front_right": (1, -1),
            "right": (0, -1),
            "back_right": (-1, -1),
            "back": (-1, 0),
            "back_left": (-1, 1),
            "left": (0, 1),
            "front_left": (1, 1),
        }

        readings = {}

        for name, (dx, dy) in directions.items():
            magnitude = math.sqrt(dx * dx + dy * dy)
            dx /= magnitude
            dy /= magnitude

            sensor_offset = self.ROBOT_RADIUS + 0.05

            start = [
                position[0] + dx * sensor_offset,
                position[1] + dy * sensor_offset,
                0.25,
            ]

            end = [
                position[0] + dx * self.SENSOR_RANGE,
                position[1] + dy * self.SENSOR_RANGE,
                0.25,
            ]

            result = p.rayTest(
                start,
                end,
                physicsClientId=self.client,
            )[0]

            if result[0] < 0:
                distance = self.SENSOR_RANGE
            else:
                distance = self.SENSOR_RANGE * result[2]

            readings[name] = round(distance, 3)

        return readings

    def move(self, direction: str) -> MoveResult:
        """Physically move the simulated robot and verify the result."""

        previous = self.agent_position

        if self._transient_move_failures:
            self._transient_move_failures -= 1
            self.events.append(
                f"move_failed:transient_failure:{previous}"
            )
            return MoveResult(False, "transient_failure", previous, previous)

        if direction not in self.DIRECTIONS:
            return self._record_move(False, "invalid_direction", previous)

        dx, dy = self.DIRECTIONS[direction]
        target = (previous[0] + dx, previous[1] + dy)

        if not self.in_bounds(target):
            return self._record_move(False, "outside_grid", previous)

        if target in self._all_obstacles():
            return self._record_move(False, "obstacle", previous)

        start_x, start_y = self._world_position(previous)
        target_x, target_y = self._world_position(target)

        max_steps = 360
        tolerance = 0.07
        reached = False

        for _ in range(max_steps):
            position, _ = p.getBasePositionAndOrientation(
                self.robot_id,
                physicsClientId=self.client,
            )

            error_x = target_x - position[0]
            error_y = target_y - position[1]
            distance = math.hypot(error_x, error_y)

            if distance <= tolerance:
                reached = True
                break

            speed = min(self.ROBOT_SPEED, distance * 3.0)

            velocity_x = speed * error_x / distance
            velocity_y = speed * error_y / distance

            p.resetBaseVelocity(
                self.robot_id,
                linearVelocity=[velocity_x, velocity_y, 0],
                angularVelocity=[0, 0, 0],
                physicsClientId=self.client,
            )

            p.stepSimulation(physicsClientId=self.client)

            for body_id in self.obstacle_bodies.values():
                contacts = p.getContactPoints(
                    bodyA=self.robot_id,
                    bodyB=body_id,
                    physicsClientId=self.client,
                )
                if contacts:
                    break
            else:
                if self.realtime:
                    time.sleep(1.0 / 240.0)
                continue

            break

        final_position, _ = p.getBasePositionAndOrientation(
            self.robot_id,
            physicsClientId=self.client,
        )

        actual_distance = math.hypot(
            target_x - final_position[0],
            target_y - final_position[1],
        )

        reached = reached or actual_distance <= tolerance

        if not reached:
            p.resetBasePositionAndOrientation(
                self.robot_id,
                [start_x, start_y, self.ROBOT_HEIGHT / 2 + 0.02],
                [0, 0, 0, 1],
                physicsClientId=self.client,
            )
            p.resetBaseVelocity(
                self.robot_id,
                [0, 0, 0],
                [0, 0, 0],
                physicsClientId=self.client,
            )
            return self._record_move(False, "physical_movement_failed", previous)

        self.agent_position = target
        self.state_version += 1
        self.events.append(f"move:{direction}:{previous}->{target}")

        return MoveResult(True, "moved", previous, target)

    def add_dynamic_obstacle(self, position, duration=None):
        super().add_dynamic_obstacle(position, duration)
        self._sync_obstacles()

    def remove_dynamic_obstacle(self, position):
        removed = super().remove_dynamic_obstacle(position)
        self._sync_obstacles()
        return removed

    def move_dynamic_obstacle(self, position, destination):
        moved = super().move_dynamic_obstacle(position, destination)
        self._sync_obstacles()
        return moved

    def advance(self):
        super().advance()
        self._sync_obstacles()
        p.stepSimulation(physicsClientId=self.client)

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(physicsClientId=self.client)
            print("REROUTE: PyBullet simulation closed.")
