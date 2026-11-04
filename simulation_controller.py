"""Thread-gated bridge between existing REROUTE logic and a live UI."""
from __future__ import annotations
from collections import deque
from threading import Condition, Thread
import time
from typing import Callable
from agent import AgentEvent, AgentExecutionCancelled, AutonomousAgent
from environment import Environment
from milestone11_scenarios import validation_factories
from visual_models import ActivityItem, DashboardSnapshot
from decision_pipeline import run_decision_pipeline
from meta_reasoning_adapter import get_recovery_recommendation
from adaptive_learning import AdaptiveOutcomeMemory


def mock_recommendation(context):
    """Offline provider for integration tests; never calls Meta."""
    if context.get("path_blocked") and context.get("safe_alternative_route"):
        action = "REPLAN"
    elif context.get("path_blocked"):
        action = "GATHER_INFORMATION"
    else:
        action = "WAIT"
    return {
        "recommended_action": action,
        "reason": "Offline test recommendation based on supplied environment state.",
        "confidence": 0.80,
        "required_checks": ["Verify the selected action against the current environment."],
        "alternative_action": "GATHER_INFORMATION" if action != "GATHER_INFORMATION" else "REQUEST_HUMAN_HELP",
        "visual_summary": "Offline provider; no image was sent to Muse Spark.",
        "uncertainties": ["This is a mock recommendation, not a model response."],
    }


def reroute_decision_policy(context, live_reasoning: bool = False):
    provider = get_recovery_recommendation if live_reasoning else mock_recommendation
    mode = "LIVE MUSE SPARK" if live_reasoning else "OFFLINE MOCK"
    print(f"REROUTE: decision pipeline triggered ({mode}).")
    result = run_decision_pipeline(context, recommendation_provider=provider)
    if result.get("status") != "EVALUATED":
        raise RuntimeError(f"Decision pipeline did not produce an executable decision: {result}")
    return result


def scenario_factories() -> dict[str, Callable[[], Environment]]:
    factories = {
        "Default navigation": Environment,
        "M4 obstacle appears": _appearing_obstacle,
        "M4 temporary obstacle": _temporary_obstacle,
        "M4 moving obstacle": _moving_obstacle,
    }
    factories.update({f"M11 {name}": factory for name, factory in validation_factories().items()})
    return factories


class SimulationController:
    """Controls playback by pausing agent callbacks, never its decisions."""
    def __init__(
        self,
        live_reasoning: bool = False,
        vision_engine=None,
        image_provider=None,
    ) -> None:
        self.live_reasoning = live_reasoning
        self.vision_engine = vision_engine
        self.image_provider = image_provider
        # Shared across simulation runs for this controller instance.
        self.adaptive_memory = AdaptiveOutcomeMemory(persistent=True)
        self._live_request_used = False
        self.factories = scenario_factories()
        self.snapshot = DashboardSnapshot()
        self.speed = 3.0
        self.paused = True
        self._step_budget = 0
        self._cancelled = False
        self._condition = Condition()
        self._thread: Thread | None = None
        self._environment: Environment | None = None
        self._agent: AutonomousAgent | None = None
        self.events: deque[AgentEvent] = deque(maxlen=200)

    @property
    def scenario_names(self) -> list[str]:
        return list(self.factories)

    def start(self, scenario: str | None = None, start_paused: bool = False) -> None:
        if self._thread and self._thread.is_alive():
            self.resume()
            return

        selected = scenario or self.snapshot.scenario
        if selected not in self.factories:
            raise ValueError(f"Unknown scenario: {selected}")

        self._cancelled = False
        # Allow one live model request per run; a new run gets a fresh allowance.
        self._live_request_used = False
        self.paused = start_paused
        self._step_budget = 0
        self.events.clear()
        self._environment = self.factories[selected]()

        def decision_policy(context):
            # Consume the single live-call allowance before making the request.
            # A failed/timeout request still counts as an attempt, preventing retries.
            use_live = self.live_reasoning and not self._live_request_used
            if use_live:
                self._live_request_used = True
            return reroute_decision_policy(context, live_reasoning=use_live)

        self._agent = AutonomousAgent(
            self._environment,
            recovery_mode="m11",
            logger=lambda _: None,
            event_callback=self._on_agent_event,
            decision_policy=decision_policy,
            vision_engine=self.vision_engine,
            image_provider=self.image_provider,
            adaptive_memory=self.adaptive_memory,
        )

        self.snapshot = DashboardSnapshot(
            scenario=selected,
            status="PAUSED" if start_paused else "RUNNING",
            state=self._environment.get_state(),
            start_position=self._environment.agent_position,
        )
        self._thread = Thread(target=self._run_agent, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        with self._condition:
            self.paused = True
            self.snapshot.status = "PAUSED"

    def resume(self) -> None:
        with self._condition:
            self.paused = False
            self.snapshot.status = "RUNNING"
            self._condition.notify_all()

    def step(self) -> None:
        with self._condition:
            if self._thread is None or not self._thread.is_alive():
                self.start(self.snapshot.scenario, start_paused=True)
            self.paused = True
            self._step_budget += 1
            self.snapshot.status = "PAUSED"
            self._condition.notify_all()

    def reset(self, scenario: str | None = None) -> None:
        with self._condition:
            self._cancelled = True
            self.paused = False
            self._condition.notify_all()
        if self._thread:
            self._thread.join(timeout=1)
        selected = scenario or self.snapshot.scenario
        if selected not in self.factories:
            raise ValueError(f"Unknown scenario: {selected}")
        environment = self.factories[selected]()
        self._environment = environment
        self._thread = None
        self.snapshot = DashboardSnapshot(scenario=selected, status="READY", state=environment.get_state(), start_position=environment.agent_position)
        self.events.clear()

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.25, min(20.0, speed))

    def _run_agent(self) -> None:
        try:
            assert self._agent is not None
            outcome = self._agent.run()
            with self._condition:
                self.snapshot.reached_goal = outcome.reached_goal
                if not outcome.reached_goal:
                    self.snapshot.safe_stop_reason = outcome.reason
                self.snapshot.status = "GOAL REACHED" if outcome.reached_goal else "SAFE STOP"
        except AgentExecutionCancelled:
            pass
        except Exception as error:
            with self._condition:
                self.snapshot.status = "ERROR"
                self.snapshot.safe_stop_reason = str(error)

    def _on_agent_event(self, event: AgentEvent) -> bool | None:
        with self._condition:
            if self._cancelled:
                return False
            while self.paused and self._step_budget == 0 and not self._cancelled:
                self._condition.wait()
            if self._cancelled:
                return False
            if self._step_budget:
                self._step_budget -= 1
            self.events.append(event)
            self._apply_event(event)
        if not self.paused:
            time.sleep(1 / self.speed)
        return True

    def _apply_event(self, event: AgentEvent) -> None:
        payload = event.payload

                # Preserve complete recovery decision and prediction history.
        if event.activity == "RECOVERING":
            decision_details = payload.get("decision_details")

            self.snapshot.decision_history.append({
                "tick": event.tick,
                "state_version": event.state_version,
                "decision_source": payload.get(
                    "decision_source", "UNKNOWN"
                ),
                "recovery_action": payload.get("recovery_action"),
                "decision_reason": payload.get("decision_reason", ""),
                "decision_error": payload.get("decision_error"),
                "decision_details": (
                    dict(decision_details)
                    if isinstance(decision_details, dict)
                    else None
                ),
            })
        self.snapshot.state = dict(payload["state"])
        if self._environment is not None:
            self.snapshot.state["obstacle_info"] = {
                tuple(position): self._environment.obstacle_info(position)
                for position in self.snapshot.state["obstacles"]
            }
        self.snapshot.route = tuple(payload.get("route", ()))
        self.snapshot.alternative_route = tuple(payload.get("alternative_route", ()))
        self.snapshot.status = event.activity
        self.snapshot.nodes_explored = int(payload.get("nodes_explored", self.snapshot.nodes_explored))
        self.snapshot.recovery_attempts = int(payload.get("recoveries", self.snapshot.recovery_attempts))
        self.snapshot.failed_movements = int(payload.get("failures", self.snapshot.failed_movements))
        message = _event_message(event)
        self.snapshot.activity.append(ActivityItem(event.activity, event.tick, message))
        self.snapshot.activity = self.snapshot.activity[-8:]
        if event.activity == "MOVING" and payload.get("expected_position"):
            self.snapshot.visited.add(tuple(payload["state"]["agent_position"]))
        failure = payload.get("failure")
        diagnosis = payload.get("diagnosis")
        if failure:
            self.snapshot.reasoning["failure"] = failure.failure_type.value
            self.snapshot.reasoning["evidence"] = failure.evidence
        if diagnosis:
            self.snapshot.reasoning["cause"] = diagnosis.cause.value
            self.snapshot.reasoning["confidence"] = f"{diagnosis.confidence:.2f}"
            self.snapshot.reasoning["support"] = diagnosis.supporting_evidence[0] if diagnosis.supporting_evidence else ""
        if payload.get("recovery_action"):
            self.snapshot.reasoning["action"] = payload["recovery_action"]
            self.snapshot.reasoning["decision"] = payload.get("decision_reason", "")
            self.snapshot.reasoning["decision_error"] = payload.get("decision_error")
            if payload["recovery_action"] == "RETRY": self.snapshot.retries += 1
            if payload["recovery_action"] == "REPLAN": self.snapshot.replans += 1
        if payload.get("verification_state"):
            self.snapshot.reasoning["verification"] = payload["verification_state"]
        if event.activity == "GOAL_REACHED": self.snapshot.reached_goal = True
        if event.activity == "SAFE_STOP": self.snapshot.safe_stop_reason = str(payload.get("reason", "safe stop"))


def _event_message(event: AgentEvent) -> str:
    payload = event.payload
    if event.activity == "MOVING": return f"{payload.get('direction', '?')} -> {payload.get('expected_position', '')}"
    if event.activity == "PLANNING": return f"{payload.get('algorithm', '')}: {payload.get('path_length', '?')} steps"
    if event.activity == "SAFE_STOP": return str(payload.get("reason", "safe stop"))
    if event.activity == "RECOVERING" and payload.get("decision_error"):
        return f"LOCAL FALLBACK: {payload['decision_error']}"
    return event.activity.replace("_", " ")


def _appearing_obstacle() -> Environment:
    environment = Environment(); environment.schedule_obstacle_event(1, "add", (0, 1)); return environment

def _temporary_obstacle() -> Environment:
    environment = Environment(); environment.schedule_obstacle_event(1, "add", (0, 1), duration=1); return environment

def _moving_obstacle() -> Environment:
    environment = Environment(); environment.add_dynamic_obstacle((1, 1)); environment.schedule_obstacle_event(1, "move", (1, 1), destination=(0, 1)); return environment
