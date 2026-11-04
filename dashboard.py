"""Live Pygame dashboard for REROUTE autonomous simulations."""

from __future__ import annotations

try:
    import pygame
except ModuleNotFoundError as error:
    raise SystemExit(
        "Pygame is required. Install it with: "
        ".\\.venv\\Scripts\\python.exe -m pip install pygame"
    ) from error

from simulation_controller import SimulationController


WINDOW = (1360, 860)
GRID_RECT = pygame.Rect(20, 70, 760, 760)
PANEL_X = 800


class Dashboard:

    def __init__(self) -> None:
        pygame.init()

        pygame.display.set_caption(
            "REROUTE | Autonomous AI Dashboard"
        )

        self.screen = pygame.display.set_mode(WINDOW)
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("consolas", 16)
        self.small = pygame.font.SysFont("consolas", 13)
        self.title = pygame.font.SysFont(
            "consolas", 22, bold=True
        )

        self.controller = SimulationController(live_reasoning=False)

        self.scenario_index = 0
        self.controller.reset(
            self.controller.scenario_names[0]
        )

        self.running = True

    def run(self) -> None:
        while self.running:
            self._handle_events()
            self._draw()

            pygame.display.flip()
            self.clock.tick(60)

        self.controller.reset()
        pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_SPACE:
                    if self.controller.paused:
                        self.controller.resume()
                    else:
                        self.controller.pause()

                elif event.key == pygame.K_s:
                    self.controller.start(
                        self.current_scenario
                    )

                elif event.key == pygame.K_r:
                    self.controller.reset(
                        self.current_scenario
                    )

                elif event.key == pygame.K_n:
                    self.controller.step()

                elif event.key == pygame.K_LEFT:
                    self.scenario_index = (
                        self.scenario_index - 1
                    ) % len(self.controller.scenario_names)

                    self.controller.reset(
                        self.current_scenario
                    )

                elif event.key == pygame.K_RIGHT:
                    self.scenario_index = (
                        self.scenario_index + 1
                    ) % len(self.controller.scenario_names)

                    self.controller.reset(
                        self.current_scenario
                    )

                elif event.key in (
                    pygame.K_MINUS,
                    pygame.K_KP_MINUS
                ):
                    self.controller.set_speed(
                        self.controller.speed - 0.5
                    )

                elif event.key in (
                    pygame.K_EQUALS,
                    pygame.K_KP_PLUS
                ):
                    self.controller.set_speed(
                        self.controller.speed + 0.5
                    )

    @property
    def current_scenario(self) -> str:
        return self.controller.scenario_names[
            self.scenario_index
        ]

    # --------------------------------------------------
    # MAIN DASHBOARD
    # --------------------------------------------------

    def _draw(self) -> None:
        self.screen.fill((16, 20, 30))

        self._text(
            "REROUTE AUTONOMOUS AI DASHBOARD",
            (20, 20),
            self.title,
            (225, 235, 255)
        )

        history = self.controller.snapshot.decision_history

        if history:
            source = history[-1].get(
                "decision_source", "UNKNOWN"
            )
        else:
            source = "OFFLINE MOCK"

        source_colors = {
            "EXTERNAL_DECISION_PIPELINE": (80, 210, 140),
            "LOCAL_RECOVERY": (245, 196, 70),
            "UNKNOWN": (180, 180, 180),
            "OFFLINE MOCK": (180, 180, 180)
        }

        source_label = {
            "EXTERNAL_DECISION_PIPELINE":
                "EXTERNAL DECISION PIPELINE",
            "LOCAL_RECOVERY": "LOCAL RECOVERY",
            "UNKNOWN": "UNKNOWN",
            "OFFLINE MOCK": "OFFLINE MOCK"
        }.get(source, str(source))

        self._text(
            f"DECISION SOURCE: {source_label}",
            (800, 25),
            self.small,
            source_colors.get(source, (180, 180, 180))
        )

        self._draw_grid()
        self._draw_activity_panel()
        self._draw_reasoning_panel()
        self._draw_stats()
        self._draw_controls()

    # --------------------------------------------------
    # NAVIGATION GRID
    # --------------------------------------------------

    def _draw_grid(self) -> None:
        snapshot = self.controller.snapshot
        state = snapshot.state

        pygame.draw.rect(
            self.screen,
            (28, 36, 52),
            GRID_RECT,
            border_radius=8
        )

        if not state:
            self._text(
                "Select a scenario and press S to start",
                (190, 430),
                self.font,
                (170, 180, 200)
            )
            return

        width = state["width"]
        height = state["height"]

        cell = min(
            GRID_RECT.width // width,
            GRID_RECT.height // height
        )

        origin_x = (
            GRID_RECT.x
            + (GRID_RECT.width - cell * width) // 2
        )

        origin_y = (
            GRID_RECT.y
            + (GRID_RECT.height - cell * height) // 2
        )

        route = set(snapshot.route)
        alternate = set(snapshot.alternative_route)

        obstacles = set(state["obstacles"])
        details = state.get("obstacle_info", {})

        for y in range(height):
            for x in range(width):

                rect = pygame.Rect(
                    origin_x + x * cell,
                    origin_y + y * cell,
                    cell,
                    cell
                )

                color = (38, 47, 66)
                position = (x, y)

                if position in snapshot.visited:
                    color = (51, 76, 105)

                if position in alternate:
                    color = (87, 65, 132)

                if position in route:
                    color = (43, 91, 132)

                if position in obstacles:
                    info = details.get(position, {})

                    if info.get("temporary"):
                        color = (235, 183, 57)

                    elif info.get("kind") == "dynamic":
                        color = (218, 119, 55)

                    else:
                        color = (83, 88, 100)

                pygame.draw.rect(
                    self.screen,
                    color,
                    rect.inflate(-2, -2),
                    border_radius=3
                )

                if (
                    position in obstacles
                    and details.get(position, {}).get("temporary")
                ):
                    self._text(
                        str(
                            details[position].get(
                                "remaining_duration", ""
                            )
                        ),
                        rect.center,
                        self.small,
                        (25, 25, 25),
                        centered=True
                    )

        if snapshot.start_position:
            self._circle_cell(
                snapshot.start_position,
                origin_x,
                origin_y,
                cell,
                (82, 210, 230),
                outline=True
            )

        self._circle_cell(
            state["goal_position"],
            origin_x,
            origin_y,
            cell,
            (63, 205, 120)
        )

        self._circle_cell(
            state["agent_position"],
            origin_x,
            origin_y,
            cell,
            (75, 160, 255)
        )

    # --------------------------------------------------
    # AGENT ACTIVITY
    # --------------------------------------------------

    def _draw_activity_panel(self) -> None:
        self._panel(
            pygame.Rect(PANEL_X, 70, 540, 240),
            "AGENT ACTIVITY"
        )

        activity = self.controller.snapshot.activity

        for index, item in enumerate(activity[-8:]):

            if item.activity in {"MOVING", "PLANNING"}:
                color = (109, 207, 255)

            elif (
                "FAILURE" in item.activity
                or "SAFE" in item.activity
            ):
                color = (249, 186, 74)

            else:
                color = (190, 220, 200)

            self._text(
                f"[{item.tick:03}] "
                f"{item.activity:<18} "
                f"{item.message[:28]}",
                (815, 105 + index * 24),
                self.small,
                color
            )

    # --------------------------------------------------
    # AI REASONING AND DECISION HISTORY
    # --------------------------------------------------

    def _draw_reasoning_panel(self) -> None:
        self._panel(
            pygame.Rect(PANEL_X, 330, 540, 240),
            "AI REASONING AND DECISION INTELLIGENCE"
        )

        snapshot = self.controller.snapshot
        reasoning = snapshot.reasoning
        history = snapshot.decision_history

        if not history:
            entries = [
                f"Failure: {reasoning.get('failure', 'None')}",
                f"Cause: {reasoning.get('cause', 'Not detected')}",
                f"Confidence: {reasoning.get('confidence', 'N/A')}",
                "Decision: Awaiting recovery event",
                "Prediction: Not available",
                "Evaluated alternatives: Not available",
                "Decision history: 0 records"
            ]

        else:
            latest = history[-1]

            details = latest.get("decision_details") or {}

            if not isinstance(details, dict):
                details = {}

            prediction = details.get("prediction") or {}

            if not isinstance(prediction, dict):
                prediction = {}

            evaluated = details.get("evaluated_actions", [])

            if not isinstance(evaluated, list):
                evaluated = []

            uncertainties = details.get("uncertainties", [])

            action = latest.get(
                "recovery_action", "UNKNOWN"
            )

            source = latest.get(
                "decision_source", "UNKNOWN"
            )

            probability = prediction.get(
                "success_probability"
            )

            if isinstance(probability, (int, float)):
                probability_text = (
                    f"{probability * 100:.1f}%"
                )
            else:
                probability_text = "N/A"

            risk = prediction.get(
                "risk_level",
                prediction.get("risk", "UNKNOWN")
            )

            feasible = prediction.get(
                "feasible", "UNKNOWN"
            )

            candidate_names = []

            for item in evaluated[:5]:

                if not isinstance(item, dict):
                    continue

                candidate_action = item.get(
                    "action", "UNKNOWN"
                )

                candidate_prediction = item.get(
                    "prediction", {}
                )

                if not isinstance(candidate_prediction, dict):
                    candidate_prediction = {}

                candidate_probability = (
                    candidate_prediction.get(
                        "success_probability"
                    )
                )

                if isinstance(
                    candidate_probability,
                    (int, float)
                ):
                    candidate_names.append(
                        f"{candidate_action}: "
                        f"{candidate_probability * 100:.0f}%"
                    )
                else:
                    candidate_names.append(
                        str(candidate_action)
                    )

            if isinstance(uncertainties, (list, tuple)):
                uncertainty_text = (
                    str(uncertainties[0])
                    if uncertainties
                    else "None recorded"
                )
            elif uncertainties:
                uncertainty_text = str(uncertainties)
            else:
                uncertainty_text = "None recorded"

            entries = [
                f"Source: {source}",
                f"Selected action: {action}",
                f"Predicted success: {probability_text}",
                f"Risk: {risk}    Feasible: {feasible}",
                "Evaluated actions: "
                + (
                    ", ".join(candidate_names)
                    if candidate_names
                    else "Not available"
                ),
                f"Uncertainty: {uncertainty_text}",
                (
                    f"History: {len(history)} decisions"
                    f"    Tick: {latest.get('tick', 0)}"
                ),
                f"Policy error: {latest.get('decision_error') or 'None'}"
            ]

        for index, value in enumerate(entries):
            color = (215, 225, 240)

            if index == 0 and history:
                color = (100, 220, 160)

            elif index == 7 and history and latest.get("decision_error"):
                color = (255, 125, 105)

            elif index == 1 and history:
                color = (110, 190, 255)

            elif index == 3 and history:
                color = (245, 196, 90)

            self._text(
                str(value)[:65],
                (815, 365 + index * 23),
                self.small,
                color
            )

        # Preserve existing diagnosis information.
        if reasoning.get("cause"):
            self._text(
                (
                    f"Diagnosis: {reasoning.get('cause')} "
                    f"({reasoning.get('confidence', 'N/A')})"
                )[:65],
                (815, 550),
                self.small,
                (170, 190, 215)
            )

    # --------------------------------------------------
    # SIMULATION STATISTICS
    # --------------------------------------------------

    def _draw_stats(self) -> None:
        self._panel(
            pygame.Rect(PANEL_X, 590, 540, 160),
            "SIMULATION STATISTICS"
        )

        snap = self.controller.snapshot
        state = snap.state

        rows = [
            f"Status: {snap.status}",

            (
                f"Tick: {state.get('tick', 0)} "
                f"Agent: {state.get('agent_position', '-')}"
                f" Goal: {state.get('goal_position', '-')}"
            ),

            (
                f"Path length: {max(0, len(snap.route) - 1)} "
                f"Nodes explored: {snap.nodes_explored}"
            ),

            (
                f"Failed movements: {snap.failed_movements} "
                f"Recoveries: {snap.recovery_attempts}"
            ),

            (
                f"Retries: {snap.retries} "
                f"Replans: {snap.replans} "
                f"Speed: {self.controller.speed:.1f} events/s"
            ),

            (
                f"Issue: {snap.safe_stop_reason or 'None'}"
            )
        ]

        for index, row in enumerate(rows):
            self._text(
                row[:75],
                (815, 625 + index * 21),
                self.small,
                (210, 220, 238)
            )

    # --------------------------------------------------
    # CONTROLS
    # --------------------------------------------------

    def _draw_controls(self) -> None:
        self._panel(
            pygame.Rect(PANEL_X, 750, 540, 80),
            "CONTROLS"
        )

        self._text(
            f"Scenario: {self.current_scenario}",
            (815, 780),
            self.small,
            (240, 210, 112)
        )

        self._text(
            "S Start | Space Pause/Resume | N Step | "
            "R Reset | Left/Right Scenario | +/- Speed",
            (815, 805),
            self.small,
            (190, 205, 225)
        )

    # --------------------------------------------------
    # DRAWING UTILITIES
    # --------------------------------------------------

    def _panel(self, rect, title) -> None:
        pygame.draw.rect(
            self.screen,
            (27, 34, 49),
            rect,
            border_radius=8
        )

        pygame.draw.rect(
            self.screen,
            (74, 92, 125),
            rect,
            1,
            border_radius=8
        )

        self._text(
            title,
            (rect.x + 14, rect.y + 12),
            self.small,
            (133, 177, 250)
        )

    def _circle_cell(
        self,
        position,
        ox,
        oy,
        cell,
        color,
        outline=False
    ) -> None:
        center = (
            ox + position[0] * cell + cell // 2,
            oy + position[1] * cell + cell // 2
        )

        pygame.draw.circle(
            self.screen,
            color,
            center,
            max(5, cell // 3),
            2 if outline else 0
        )

    def _text(
        self,
        text,
        position,
        font,
        color,
        centered=False
    ) -> None:
        surface = font.render(
            str(text),
            True,
            color
        )

        if centered:
            rect = surface.get_rect(center=position)
        else:
            rect = surface.get_rect(topleft=position)

        self.screen.blit(surface, rect)


if __name__ == "__main__":
    Dashboard().run()