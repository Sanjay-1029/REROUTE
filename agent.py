"""Autonomous route planning and verified movement execution for REROUTE."""

from __future__ import annotations

from dataclasses import dataclass, field
import base64
import os
import time
from typing import Any, Callable

from environment import Environment, MoveResult, Position
from diagnosis import FailureDiagnosis
from failure_detector import FailureDetector, FailureRecord
from pathfinder import SearchResult, astar_search, bfs_search
from recovery import RecoveryAction, RecoveryEngine, RecoveryRecord
from outcome_verifier import VerificationState
from adaptive_learning import AdaptiveOutcomeMemory
from recovery import RecoveryDecision, RecoveryAction



@dataclass(frozen=True)
class ActionRecord:
    step: int
    direction: str | None
    expected_position: Position | None
    actual_position: Position
    succeeded: bool
    reason: str


@dataclass(frozen=True)
class AgentEvent:
    """A copy-safe observation emitted for dashboards and other observers."""

    activity: str
    tick: int
    state_version: int
    payload: dict[str, Any]


class AgentExecutionCancelled(Exception):
    """Raised by an observer that intentionally stops a live simulation."""


@dataclass
class AgentRunResult:
    reached_goal: bool
    reason: str
    planned_route: list[list[int]] | None
    actions: list[ActionRecord] = field(default_factory=list)
    invalidated_positions: list[Position] = field(default_factory=list)
    failures: list[FailureRecord] = field(default_factory=list)
    recoveries: list[RecoveryRecord] = field(default_factory=list)

    @property
    def successful_moves(self) -> int:
        return sum(action.succeeded for action in self.actions)

    @property
    def failed_moves(self) -> int:
        return len(self.actions) - self.successful_moves


class AutonomousAgent:
    """Plans with BFS/A* and terminates safely when verified progress stops."""

    _DIRECTION_FOR_DELTA = {(0, -1): "up", (0, 1): "down", (-1, 0): "left", (1, 0): "right"}

    def __init__(
        self,
        environment: Environment,
        algorithm: str = "astar",
        logger: Callable[[str], None] = print,
        max_recovery_attempts: int = 3,
        recovery_enabled: bool = True,
        recovery_mode: str = "m9",
        max_wait_ticks: int = 5,
        event_callback: Callable[[AgentEvent], bool | None] | None = None,
        decision_policy: Callable[[dict], dict] | None = None,
        vision_engine: Any | None = None,
        image_provider: Callable[[], Any] | None = None,
        adaptive_memory: AdaptiveOutcomeMemory | None = None,
    ) -> None:
        if algorithm not in {"bfs", "astar"}:
            raise ValueError("algorithm must be 'bfs' or 'astar'.")
        if recovery_mode not in {"m9", "m11"}:
            raise ValueError("recovery_mode must be 'm9' or 'm11'.")
        self.environment = environment
        self.algorithm = algorithm
        self.logger = logger
        self.route: list[list[int]] | None = None
        self.actions: list[ActionRecord] = []
        self.invalidated_positions: list[Position] = []
        self.failure_detector = FailureDetector()
        self.diagnosis_engine = FailureDiagnosis()
        self.recovery_engine = RecoveryEngine(max_recovery_attempts)
        self.recovery_enabled = recovery_enabled
        self.recovery_mode = recovery_mode
        self.max_wait_ticks = max_wait_ticks
        self._pending_recoveries: list[str] = []
        self._movement_limit = 0
        self.event_callback = event_callback
        self.decision_policy = decision_policy
        self.vision_engine = vision_engine
        self.image_provider = image_provider
        self.adaptive_memory = adaptive_memory or AdaptiveOutcomeMemory()
        self.alternative_route: list[list[int]] | None = None
        self.last_visual_context: dict[str, Any] = {}
        self.visual_observation_history: list[dict[str, Any]] = []

    def plan_route(self) -> SearchResult:
        """Read a fresh environment snapshot and plan from the actual position."""
        self._emit("OBSERVING")
        state = self.environment.get_state()
        search = astar_search if self.algorithm == "astar" else bfs_search
        result = search(
            state["agent_position"], state["goal_position"], state["obstacles"],
            state["width"], state["height"],
        )
        self.route = result.path
        self._emit("PLANNING", algorithm=result.algorithm, nodes_explored=result.explored_nodes, path_length=result.steps)
        if result.path:
            self._log(f"PLAN {result.algorithm}: {result.steps} steps, {result.explored_nodes} nodes explored.")
        else:
            self._log(f"PLAN {result.algorithm}: no route available.")
        return result

    def run(self, max_steps: int | None = None) -> AgentRunResult:
        """Execute verified routes and bounded, diagnosis-driven recovery actions."""
        self.actions = []
        self.invalidated_positions = []
        self.failure_detector = FailureDetector()
        self.diagnosis_engine = FailureDiagnosis()
        self.recovery_engine = RecoveryEngine(self.recovery_engine.max_attempts)
        self._pending_recoveries = []
        self.last_visual_context = {}
        self.visual_observation_history = []
        state = self.environment.get_state()
        if state["agent_position"] == state["goal_position"]:
            self.failure_detector.observe_goal_reached(self.environment)
            self._log("GOAL already reached.")
            self._emit("GOAL_REACHED")
            return self._result(True, "goal_reached")

        limit = max_steps if max_steps is not None else state["width"] * state["height"]
        self._movement_limit = limit
        while len(self.actions) < limit:
            search_result = self.plan_route()
            if not search_result.path:
                failure = self.failure_detector.observe_no_route(self.environment)
                if not self.recovery_enabled:
                    return self._result(False, "no_route")
                if not self._recover(failure, None):
                    return self._result(False, "no_route")
                continue
            recovery_needed = False
            for index in range(1, len(search_result.path)):
                if len(self.actions) >= limit:
                    return self._result(False, "step_limit_reached")
                self.environment.advance()
                blocked = self._blocked_route_positions(search_result.path[index:])
                if blocked:
                    self.invalidated_positions = blocked
                    failure = self.failure_detector.observe_route_outdated(self.environment, blocked)
                    self._emit("FAILURE_DETECTED", failure=failure, affected_nodes=blocked)
                    self._log(f"ROUTE OUTDATED: obstacle(s) now occupy {blocked}.")
                    if not self.recovery_enabled:
                        return self._result(False, "route_outdated")
                    if not self._recover(failure, tuple(search_result.path[index])):
                        return self._result(False, "route_outdated")
                    recovery_needed = True
                    break
                record = self.execute_next(search_result.path[index])
                if not record.succeeded:
                    failure = self.failure_detector.history[-1]
                    self._emit("FAILURE_DETECTED", failure=failure)
                    if not self.recovery_enabled:
                        return self._result(False, record.reason)
                    if not self._recover(failure, tuple(search_result.path[index])):
                        return self._result(False, record.reason)
                    recovery_needed = True
                    break
                if self.environment.agent_position == self.environment.goal_position:
                    self.failure_detector.observe_goal_reached(self.environment)
                    self._verify_task_completed()
                    self._log("GOAL reached and position verified.")
                    self._emit("GOAL_REACHED")
                    return self._result(True, "goal_reached")
            if not recovery_needed:
                return self._result(False, "route_finished_without_goal")
        return self._result(False, "step_limit_reached")

    def execute_next(self, expected_position: Position | list[int]) -> ActionRecord:
        """Select, execute, and verify the one action that should reach a route node."""
        expected = tuple(expected_position)
        current = self.environment.agent_position
        delta = (expected[0] - current[0], expected[1] - current[1])
        direction = self._DIRECTION_FOR_DELTA.get(delta)
        step = len(self.actions) + 1
        if direction is None:
            record = ActionRecord(step, None, expected, current, False, "invalid_route_step")
            self.actions.append(record)
            self._log(f"STEP {step}: invalid route transition {current} -> {expected}.")
            return record

        self._log(f"STEP {step}: at {current}; choosing {direction} toward {expected}.")
        self._emit("MOVING", direction=direction, expected_position=expected)
        movement: MoveResult = self.environment.move(direction)
        actual = self.environment.agent_position
        if not movement.moved:
            reason = f"movement_blocked:{movement.reason}"
            succeeded = False
        elif actual != expected:
            reason = "unexpected_position"
            succeeded = False
        else:
            reason = "moved"
            succeeded = True
        record = ActionRecord(step, direction, expected, actual, succeeded, reason)
        self.actions.append(record)
        detected = self.failure_detector.observe_movement(self.environment, movement, expected)
        if detected:
            self._log(f"CLASSIFIED {detected.failure_type.value}: {detected.evidence}")
        self._log(f"STEP {step}: {'verified' if succeeded else 'failure'}; actual position {actual} ({reason}).")
        if succeeded and self.recovery_mode == "m11" and self._pending_recoveries:
            self._verify_progress_restored(actual)
        return record

    def _result(self, reached_goal: bool, reason: str) -> AgentRunResult:
        if self.recovery_mode == "m11" and not reached_goal:
            for attempt_id in self._pending_recoveries:
                self.recovery_engine.verify(attempt_id, VerificationState.RECOVERY_FAILED, f"Task ended before restored progress: {reason}.")
        self.adaptive_memory.ingest(self.recovery_engine.history)
        if not reached_goal:
            self._emit("SAFE_STOP", reason=reason)
        return AgentRunResult(reached_goal, reason, self.route, list(self.actions), list(self.invalidated_positions), list(self.failure_detector.history), list(self.recovery_engine.history))

    def _recover(self, failure: FailureRecord, expected: Position | None) -> bool:
        if self.recovery_mode == "m11":
            return self._recover_m11(failure, expected)
        return self._recover_m9(failure, expected)

    def _recover_m9(self, failure: FailureRecord, expected: Position | None) -> bool:
        diagnosis = self.diagnosis_engine.diagnose(failure, self.environment, self.actions, self.failure_detector.history)
        self._emit("DIAGNOSING", failure=failure, diagnosis=diagnosis)
        alternative = self._safe_alternative_exists()
        decision = self.recovery_engine.decide(diagnosis, self.environment, alternative)
        self._emit("RECOVERING", diagnosis=diagnosis, recovery_action=decision.action.value, decision_reason=decision.reason)
        self._log(f"RECOVERY {decision.action.value}: {decision.reason}")
        if decision.action == RecoveryAction.REQUEST_HUMAN_HELP:
            self.recovery_engine.record(diagnosis, decision, "human_help_requested", False)
            return False
        if decision.action == RecoveryAction.WAIT:
            self.environment.advance()
            obstacle = self.environment.obstacle_info(expected) if expected else {"present": True}
            success = not obstacle.get("present", False)
            self.recovery_engine.record(diagnosis, decision, "obstacle_cleared" if success else "condition_persists", success, ticks_spent=1)
            return success
        if decision.action == RecoveryAction.RETRY and expected is not None:
            result = self.execute_next(expected)
            retry_count = 1
            while (
                not result.succeeded
                and retry_count < 2
                and "transient_failure" in result.reason
                and self._safe_alternative_exists()
            ):
                retry_count += 1
                result = self.execute_next(expected)
            self.recovery_engine.record(diagnosis, decision, f"{result.reason}; verified_retries={retry_count}", result.succeeded)
            return result.succeeded
        if decision.action == RecoveryAction.GATHER_INFORMATION:
            snapshot = self.environment.get_state()
            self.recovery_engine.record(diagnosis, decision, f"snapshot_v{snapshot['state_version']}", True)
            return True
        if decision.action == RecoveryAction.REPLAN:
            self.recovery_engine.record(diagnosis, decision, "alternative_route_available" if alternative else "no_alternative_route", alternative)
            return alternative
        self.recovery_engine.record(diagnosis, decision, "recovery_not_executable", False)
        return False

    def _collect_visual_context(
        self,
        failure: FailureRecord,
        diagnosis: FailureDiagnosis,
        alternative: bool,
    ) -> dict[str, Any]:
        """Acquire a fresh frame, run local candidate detection, and preserve image provenance.

        A provider may return an image path or an OpenCV/numpy image array.
        YOLO observations remain candidates; they are never converted into map
        coordinates or confirmed obstacles.
        """
        context: dict[str, Any] = {
            "perception_source": "YOLO_AND_MUSE_SPARK",
            "visual_observations": [],
            "visual_detection_count": 0,
            "image_attached_by_reasoning_adapter": False,
            "perception_status": "NO_IMAGE_CONFIGURED",
            "visual_note": "No fresh image was acquired.",
        }

        image_value = None
        if self.image_provider is not None:
            try:
                image_value = self.image_provider()
            except Exception as error:
                context.update({
                    "perception_status": "IMAGE_ACQUISITION_FAILED",
                    "visual_note": f"Image provider failed: {error}",
                })
                return context
        else:
            configured_path = os.getenv("REROUTE_IMAGE_PATH")
            if configured_path and os.path.isfile(os.path.expanduser(configured_path)):
                image_value = os.path.expanduser(configured_path)

        if image_value is None:
            return context

        try:
            if isinstance(image_value, (str, os.PathLike)):
                image_path = os.fspath(image_value)
                if not os.path.isfile(os.path.expanduser(image_path)):
                    raise FileNotFoundError("Image provider returned a missing image path.")
                if self.vision_engine is not None:
                    image = self.vision_engine.load_image(image_path)
                else:
                    image = None
                context["image_path"] = os.path.expanduser(image_path)
            else:
                image = image_value
                image_path = None

            if image is not None and self.vision_engine is not None:
                visual_state = self.vision_engine.prepare_reasoning_context(
                    {
                        "failure_type": failure.failure_type.value,
                        "diagnosis": diagnosis.cause.value,
                        "confidence": diagnosis.confidence,
                        "safe_alternative_route": alternative,
                        "environment_state": self.environment.get_state(),
                    },
                    image,
                )
                context.update(visual_state)

            # Encode in-memory frames for the existing multimodal adapter.
            if image is not None and image_path is None:
                import cv2
                import numpy as np
                if not isinstance(image, np.ndarray):
                    raise TypeError("image_provider must return an image path or numpy image array.")
                ok, encoded = cv2.imencode(".jpg", image)
                if not ok:
                    raise ValueError("Could not encode acquired image for Muse Spark.")
                context["image_data_url"] = (
                    "data:image/jpeg;base64,"
                    + base64.b64encode(encoded.tobytes()).decode("ascii")
                )

            if image_path is not None:
                context["perception_status"] = "IMAGE_READY_FOR_MUSE_SPARK"
                context["perception_source"] = "YOLO_AND_MUSE_SPARK" if self.vision_engine else "META_MUSE_SPARK_DIRECT_IMAGE_INPUT"
            elif image is not None:
                context["perception_status"] = "IMAGE_READY_FOR_MUSE_SPARK"
                context["perception_source"] = "YOLO_AND_MUSE_SPARK" if self.vision_engine else "META_MUSE_SPARK_DIRECT_IMAGE_INPUT"

            context["image_attached_by_reasoning_adapter"] = True
            context["visual_note"] = (
                "Fresh image acquired. YOLO outputs are unverified visual candidates; "
                "image-relative positions are not physical distances or grid coordinates."
            )
            return context
        except Exception as error:
            # A failed perception step must not be presented as valid visual evidence.
            context.update({
                "perception_status": "IMAGE_PROCESSING_FAILED",
                "perception_source": "VISUAL_ACQUISITION",
                "visual_observations": [],
                "visual_detection_count": 0,
                "image_attached_by_reasoning_adapter": False,
                "image_path": None,
                "image_data_url": None,
                "visual_note": f"Image processing failed: {error}",
            })
            return context

    def _record_visual_observation(self, context: dict[str, Any], phase: str) -> dict[str, Any]:
        """Append an auditable observation-cycle entry without promoting pixels to map facts."""
        state = self.environment.get_state()
        entry = {
            "sequence": len(self.visual_observation_history) + 1,
            "phase": phase,
            "tick": state["tick"],
            "state_version": state["state_version"],
            "captured_at_monotonic_ns": time.monotonic_ns(),
            "perception_status": context.get("perception_status", "UNKNOWN"),
            "perception_source": context.get("perception_source", "UNKNOWN"),
            "visual_detection_count": int(context.get("visual_detection_count", 0) or 0),
            "image_available": context.get("perception_status") == "IMAGE_READY_FOR_MUSE_SPARK",
            "visual_observations": list(context.get("visual_observations", [])),
            "visual_note": context.get("visual_note", ""),
        }
        self.visual_observation_history.append(entry)
        self._emit(
            "OBSERVING",
            observation_phase=phase,
            observation=entry,
        )
        return entry

    def _recover_m11(self, failure: FailureRecord, expected: Position | None) -> bool:
        """Candidate recovery with visual context and bounded verification."""

        diagnosis = self.diagnosis_engine.diagnose(
            failure,
            self.environment,
            self.actions,
            self.failure_detector.history
        )

        self._emit("DIAGNOSING", failure=failure, diagnosis=diagnosis)

        alternative = self._safe_alternative_exists()

        # Collect visual observations independently of the external policy.
        visual_context = self._collect_visual_context(
            failure=failure,
            diagnosis=diagnosis,
            alternative=alternative
        )

        self.last_visual_context = visual_context
        self._record_visual_observation(visual_context, "PRE_DECISION")

        self._log(
            f"VISUAL PERCEPTION: "
            f"{visual_context.get('perception_status', 'UNKNOWN')} | "
            f"detections={visual_context.get('visual_detection_count', 0)}"
        )
        decision_data = None
        decision_source = "LOCAL_RECOVERY"
        decision_error = None

        # Only verified prior outcomes enter adaptive memory. Ingest before
        # each new decision so completed recoveries can inform later choices.
        self.adaptive_memory.ingest(self.recovery_engine.history)
        adaptive_snapshot = self.adaptive_memory.snapshot(diagnosis.cause.value)

        # Use external decision policy if configured.
        if self.decision_policy is not None:
            try:
                decision_data = self.decision_policy({
                    "failure_type": failure.failure_type.value,
                    "diagnosis": diagnosis.cause.value,
                    "confidence": diagnosis.confidence,
                    "environment_state": self.environment.get_state(),
                    "safe_alternative_route": alternative,
                    "obstacles_known": True,
                    "path_blocked": failure.failure_type.value == "MOVEMENT_BLOCKED",
                    "recovery_history": [
                        record.decision.action.value
                        for record in self.recovery_engine.history
                    ],
                    "visual_context": visual_context,
                    "visual_observations": visual_context.get(
                        "visual_observations", []
                    ),
                    "visual_detection_count": visual_context.get(
                        "visual_detection_count", 0
                    ),
                    "perception_status": visual_context.get(
                        "perception_status", "UNKNOWN"
                    ),
                    "image_path": visual_context.get("image_path"),
                    "image_data_url": visual_context.get("image_data_url"),
                    "perception_source": visual_context.get("perception_source"),
                    "visual_note": visual_context.get("visual_note"),
                    "visual_navigation_warning": visual_context.get("visual_navigation_warning", False),
                    "perception_limitation": visual_context.get("perception_limitation"),
                    "adaptive_learning": adaptive_snapshot,
                })

                if not isinstance(decision_data, dict):
                    raise TypeError("Decision policy must return a dictionary.")

                if "selected_action" not in decision_data:
                    raise KeyError(
                        "Decision policy result is missing 'selected_action'."
                    )

                action = RecoveryAction(decision_data["selected_action"])

                decision = RecoveryDecision(
                    action=action,
                    reason="Decision pipeline recommendation"
                )
                decision_source = "EXTERNAL_DECISION_PIPELINE"

            except Exception as error:
                decision_error = str(error)
                self._log(
                    f"External decision failed: {decision_error}. Using local recovery."
                )

                # Do not display an incomplete or invalid result as a model decision.
                decision_data = None
                decision_source = "LOCAL_RECOVERY"

                decision = self.recovery_engine.decide(
                    diagnosis,
                    self.environment,
                    alternative
                )

        else:
            decision = self.recovery_engine.decide(
                diagnosis,
                self.environment,
                alternative
            )

        start_tick = self.environment.tick
        start_version = self.environment.state_version
        nodes = (expected,) if expected else ()

        self._log(
            f"RECOVERY {decision.action.value}: {decision.reason}"
        )

        # Forward the complete pipeline result with the actual selected action.
        # The observer receives this data but does not control agent execution.
        self._emit(
            "RECOVERING",
            diagnosis=diagnosis,
            recovery_action=decision.action.value,
            decision_reason=decision.reason,
            decision_source=decision_source,
            decision_details=decision_data,
            decision_error=decision_error
        )

        if decision.action == RecoveryAction.REQUEST_HUMAN_HELP:
            self.recovery_engine.record(
                diagnosis, decision, "human_help_requested", False,
                start_tick=start_tick,
                end_tick=self.environment.tick,
                affected_route_nodes=nodes,
                observation_versions=(
                    start_version,
                     self.environment.state_version
                )
            )
            return False

        if decision.action == RecoveryAction.WAIT:
            info = (
                self.environment.obstacle_info(expected)
                if expected else {}
            )

            duration = info.get("remaining_duration")

            budget = min(
                self.max_wait_ticks,
                duration if isinstance(duration, int)
                else self.max_wait_ticks
            )

            budget = max(1, budget)
            versions = [start_version]

            for waited in range(1, budget + 1):
                self.environment.advance()
                versions.append(self.environment.state_version)

                info = (
                    self.environment.obstacle_info(expected)
                    if expected else {"present": True}
                )

                if not info.get("present", False):
                    record = self.recovery_engine.record(
                        diagnosis, decision, "obstacle_cleared", True,
                        waited, start_tick, self.environment.tick,
                        nodes, tuple(versions)
                    )

                    self.recovery_engine.verify(
                        record.attempt_id,
                        VerificationState.ENVIRONMENT_CHANGED,
                        "Expected route node became clear during bounded wait."
                    )

                    self._pending_recoveries.append(record.attempt_id)
                    return True

                if self._safe_alternative_exists():
                    record = self.recovery_engine.record(
                        diagnosis, decision, "alternative_route_observed",
                        True, waited, start_tick, self.environment.tick,
                        nodes, tuple(versions)
                    )

                    self._pending_recoveries.append(record.attempt_id)
                    return True

            self.recovery_engine.record(
                diagnosis, decision, "wait_budget_exhausted", False,
                budget, start_tick, self.environment.tick,
                nodes, tuple(versions)
            )
            return False

        if decision.action == RecoveryAction.RETRY and expected is not None:
            versions = [start_version]
            result = None

            for retry_count in range(1, 3):
                if len(self.actions) >= self._movement_limit:
                    break

                self.environment.advance()
                versions.append(self.environment.state_version)

                if expected in self._blocked_route_positions([list(expected)]):
                    break

                result = self.execute_next(expected)

                if result.succeeded:
                    record = self.recovery_engine.record(
                        diagnosis, decision,
                        f"retry_succeeded; verified_retries={retry_count}",
                        True, retry_count, start_tick,
                        self.environment.tick, nodes, tuple(versions)
                    )

                    self._pending_recoveries.append(record.attempt_id)
                    return True

                if "transient_failure" not in result.reason:
                    break

            self.recovery_engine.record(
                diagnosis, decision,
                "retry_budget_exhausted_or_route_changed",
               False, 0, start_tick, self.environment.tick,
               nodes, tuple(versions)
            )
            return False

        if decision.action == RecoveryAction.GATHER_INFORMATION:
            # The action is not complete until a post-action observation has
            # been requested. The provider is called again so camera-backed
            # implementations can return a genuinely new frame.
            snapshot = self.environment.get_state()
            refreshed_context = self._collect_visual_context(
                failure=failure,
                diagnosis=diagnosis,
                alternative=self._safe_alternative_exists(),
            )
            self.last_visual_context = refreshed_context
            observation = self._record_visual_observation(
                refreshed_context,
                "POST_ACTION_REOBSERVATION",
            )
            refreshed = observation["image_available"]
            result_text = (
                f"snapshot_v{snapshot['state_version']}; "
                f"post_action_observation={observation['perception_status']}"
            )

            record = self.recovery_engine.record(
                diagnosis, decision,
                result_text,
                True, 0, start_tick, self.environment.tick,
                nodes,
                (start_version, snapshot["state_version"])
            )
            if refreshed:
                self.recovery_engine.verify(
                    record.attempt_id,
                    VerificationState.OBSERVATION_REFRESHED,
                    (
                        "A post-action image was acquired and passed through "
                        "the configured local visual-perception path."
                    ),
                )

            self._pending_recoveries.append(record.attempt_id)
            return True

        if decision.action == RecoveryAction.REPLAN:
            record = self.recovery_engine.record(
                diagnosis, decision,
                "alternative_route_available"
                if alternative else "no_alternative_route",
                alternative, 0, start_tick, self.environment.tick,
                nodes,
                (start_version, self.environment.state_version)
            )

            if alternative:
                self._pending_recoveries.append(record.attempt_id)

            return alternative

        self.recovery_engine.record(
            diagnosis, decision, "recovery_not_executable", False,
            start_tick=start_tick,
            end_tick=self.environment.tick,
            affected_route_nodes=nodes,
            observation_versions=(
                start_version,
                self.environment.state_version
            )
        )

        return False

    def _verify_progress_restored(self, position: Position) -> None:
        for attempt_id in self._pending_recoveries:
            self.recovery_engine.verify(attempt_id, VerificationState.PROGRESS_RESTORED, f"Navigation resumed with verified movement to {position}.")
        self._pending_recoveries = []
        self._emit("VERIFYING", verification_state=VerificationState.PROGRESS_RESTORED.value, position=position)

    def _verify_task_completed(self) -> None:
        for record in self.recovery_engine.history:
            if VerificationState.PROGRESS_RESTORED in record.verification_states:
                self.recovery_engine.verify(record.attempt_id, VerificationState.TASK_COMPLETED, "Agent position equals the goal position.")
        self._emit("VERIFYING", verification_state=VerificationState.TASK_COMPLETED.value)

    def _safe_alternative_exists(self) -> bool:
        state = self.environment.get_state()
        result = (astar_search if self.algorithm == "astar" else bfs_search)(state["agent_position"], state["goal_position"], state["obstacles"], state["width"], state["height"])
        self.alternative_route = result.path
        return bool(result.path)

    def _blocked_route_positions(self, route: list[list[int]]) -> list[Position]:
        state = self.environment.get_state()
        obstacles = set(state["obstacles"])
        return [tuple(position) for position in route if tuple(position) in obstacles]

    def _log(self, message: str) -> None:
        self.logger(f"[Agent] {message}")

    def _emit(self, activity: str, **payload: Any) -> None:
        """Notify an optional observer without giving it control of agent decisions."""
        if self.event_callback is None:
            return
        state = self.environment.get_state()
        event_payload = {
            "state": state,
            "route": tuple(tuple(position) for position in self.route or ()),
            "alternative_route": tuple(tuple(position) for position in self.alternative_route or ()),
            "actions": len(self.actions),
            "failures": len(self.failure_detector.history),
            "recoveries": len(self.recovery_engine.history),
            **payload,
        }
        event = AgentEvent(activity, state["tick"], state["state_version"], event_payload)
        if self.event_callback(event) is False:
            raise AgentExecutionCancelled("Simulation observer cancelled execution.")
