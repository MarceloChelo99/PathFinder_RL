"""Pygame renderer for interactive simulation display and control."""

import time


class PygameRenderer:
    """
    Interactive window renderer with controls:
      - SPACE: pause/resume simulation
      - N: advance one step when paused
      - ESC / close button: stop run
      - Click checkbox: toggle normalized quadrant means overlay text
    """

    def __init__(self, env, show_pheromone=True, scale_pheromone=1.0, cell_size=36):
        try:
            import pygame
        except ImportError as exc:
            raise ImportError(
                "pygame is required for visualization. Install it with `pip install pygame`."
            ) from exc

        self.pygame = pygame
        self.env = env
        self.show_pheromone = show_pheromone
        self.scale_pheromone = max(1e-6, float(scale_pheromone))
        self.cell_size = cell_size
        self.running = True
        self.paused = False

        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont("consolas", 18)
        self.small_font = pygame.font.SysFont("consolas", 16)

        self.info_h = 170
        width = env.W * cell_size
        height = env.H * cell_size + self.info_h
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("PathFinder RL Simulation")

        self.show_quad_means = False
        self.checkbox_rect = pygame.Rect(10, env.H * cell_size + 118, 18, 18)

    def _handle_event(self, event):
        if event.type == self.pygame.QUIT:
            self.running = False
        elif event.type == self.pygame.KEYDOWN:
            if event.key == self.pygame.K_ESCAPE:
                self.running = False
            elif event.key == self.pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key == self.pygame.K_n and self.paused:
                return "step"
        elif event.type == self.pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.checkbox_rect.collidepoint(event.pos):
                self.show_quad_means = not self.show_quad_means
        return None

    def _process_events(self):
        step_once = False
        for event in self.pygame.event.get():
            action = self._handle_event(event)
            if action == "step":
                step_once = True
        return step_once

    def _pheromone_purple(self, p):
        # low pheromone -> light purple, high -> dark purple
        p = max(0.0, min(1.0, p / self.scale_pheromone))
        low = (228, 204, 255)
        high = (86, 24, 130)
        return tuple(int(low[i] + (high[i] - low[i]) * p) for i in range(3))

    def _draw_grid(self):
        pygame = self.pygame
        for y in range(self.env.H):
            for x in range(self.env.W):
                tile = self.env.grid[y][x]
                if tile == "1":
                    # Obstacles are always brown.
                    color = (139, 94, 60)
                else:
                    if self.show_pheromone and hasattr(self.env, "P"):
                        color = self._pheromone_purple(self.env.P[y][x])
                    else:
                        color = (200, 200, 210)

                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (30, 30, 30), rect, 1)

        # Highlight the 3x3 local observation box around the agent and its quadrant splits.
        ax, ay = self.env.pos
        x0 = max(0, ax - 1)
        y0 = max(0, ay - 1)
        x1 = min(self.env.W - 1, ax + 1)
        y1 = min(self.env.H - 1, ay + 1)
        left = x0 * self.cell_size
        top = y0 * self.cell_size
        width = (x1 - x0 + 1) * self.cell_size
        height = (y1 - y0 + 1) * self.cell_size
        pygame.draw.rect(self.screen, (255, 255, 0), pygame.Rect(left, top, width, height), 3)

        # Crosshair through the agent to emphasize local quadrant averaging areas.
        cx = ax * self.cell_size + self.cell_size // 2
        cy = ay * self.cell_size + self.cell_size // 2
        pygame.draw.line(self.screen, (255, 230, 80), (cx, top), (cx, top + height), 2)
        pygame.draw.line(self.screen, (255, 230, 80), (left, cy), (left + width, cy), 2)

        gx, gy = self.env.goal
        goal_rect = pygame.Rect(gx * self.cell_size, gy * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (51, 220, 51), goal_rect)

        center = (ax * self.cell_size + self.cell_size // 2, ay * self.cell_size + self.cell_size // 2)
        pygame.draw.circle(self.screen, (0, 0, 0), center, self.cell_size // 3)

    def _draw_status(self, title, subtitle="", detail_lines=None):
        pygame = self.pygame
        y0 = self.env.H * self.cell_size
        pygame.draw.rect(self.screen, (20, 20, 20), pygame.Rect(0, y0, self.env.W * self.cell_size, self.info_h))

        controls = "[SPACE] pause/resume  [N] step while paused  [ESC] quit"
        status = f"{title}"
        if self.paused:
            status = f"PAUSED | {status}"

        text1 = self.font.render(status[:220], True, (240, 240, 240))
        text2 = self.small_font.render(subtitle[:220], True, (205, 205, 205))
        text3 = self.small_font.render(controls, True, (180, 180, 180))
        self.screen.blit(text1, (8, y0 + 6))
        self.screen.blit(text2, (8, y0 + 32))
        self.screen.blit(text3, (8, y0 + 56))

        if detail_lines:
            for idx, line in enumerate(detail_lines[:3]):
                text = self.small_font.render(line[:220], True, (160, 210, 255))
                self.screen.blit(text, (8, y0 + 78 + idx * 20))

        pygame.draw.rect(self.screen, (220, 220, 220), self.checkbox_rect, 1)
        if self.show_quad_means:
            pygame.draw.line(self.screen, (220, 220, 220), self.checkbox_rect.topleft, self.checkbox_rect.bottomright, 2)
            pygame.draw.line(self.screen, (220, 220, 220), self.checkbox_rect.topright, self.checkbox_rect.bottomleft, 2)

        check_text = self.small_font.render(
            "Show normalized 2x2 quadrant means (UL, UR, DR, DL)",
            True,
            (210, 210, 210),
        )
        self.screen.blit(check_text, (self.checkbox_rect.right + 8, self.checkbox_rect.y + 1))

    def update(self, title="", subtitle="", detail_lines=None):
        if not self.running:
            return False

        step_once = self._process_events()
        while self.paused and self.running and not step_once:
            self._draw_grid()
            self._draw_status(title, subtitle, detail_lines=detail_lines)
            self.pygame.display.flip()
            time.sleep(0.01)
            step_once = self._process_events()

        if not self.running:
            return False

        self._draw_grid()
        self._draw_status(title, subtitle, detail_lines=detail_lines)
        self.pygame.display.flip()
        return True

    def close(self):
        if self.running:
            self.running = False
        self.pygame.quit()
