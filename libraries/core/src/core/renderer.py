"""Pygame renderer for interactive simulation display and control."""

import time


class PygameRenderer:
    """
    Interactive window renderer with controls:
      - SPACE: pause/resume simulation
      - N: advance one step when paused
      - ESC / close button: stop run
    """

    def __init__(self, env, show_pheromone=True, scale_pheromone=1.0, cell_size=48):
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

        self.info_h = 64
        width = env.W * cell_size
        height = env.H * cell_size + self.info_h
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("PathFinder RL Simulation")

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
        return None

    def _process_events(self):
        step_once = False
        for event in self.pygame.event.get():
            action = self._handle_event(event)
            if action == "step":
                step_once = True
        return step_once

    def _draw_grid(self):
        pygame = self.pygame
        for y in range(self.env.H):
            for x in range(self.env.W):
                tile = self.env.grid[y][x]
                color = (51, 128, 255) if tile == "0" else (255, 76, 76)

                if self.show_pheromone and hasattr(self.env, "P"):
                    p = max(0.0, min(1.0, self.env.P[y][x] / self.scale_pheromone))
                    dim = int(153 * p)
                    color = tuple(max(0, c - dim) for c in color)

                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (30, 30, 30), rect, 1)

        gx, gy = self.env.goal
        goal_rect = pygame.Rect(gx * self.cell_size, gy * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (51, 255, 51), goal_rect)

        ax, ay = self.env.pos
        center = (ax * self.cell_size + self.cell_size // 2, ay * self.cell_size + self.cell_size // 2)
        pygame.draw.circle(self.screen, (0, 0, 0), center, self.cell_size // 3)

    def _draw_status(self, title):
        pygame = self.pygame
        y0 = self.env.H * self.cell_size
        pygame.draw.rect(self.screen, (20, 20, 20), pygame.Rect(0, y0, self.env.W * self.cell_size, self.info_h))

        controls = "[SPACE] pause/resume  [N] step when paused  [ESC] quit"
        status = f"{title}"
        if self.paused:
            status = f"PAUSED | {status}"

        text1 = self.font.render(status[:180], True, (240, 240, 240))
        text2 = self.font.render(controls, True, (180, 180, 180))
        self.screen.blit(text1, (8, y0 + 8))
        self.screen.blit(text2, (8, y0 + 34))

    def update(self, title=""):
        if not self.running:
            return False

        step_once = self._process_events()
        while self.paused and self.running and not step_once:
            self._draw_grid()
            self._draw_status(title)
            self.pygame.display.flip()
            time.sleep(0.01)
            step_once = self._process_events()

        if not self.running:
            return False

        self._draw_grid()
        self._draw_status(title)
        self.pygame.display.flip()
        return True

    def close(self):
        if self.running:
            self.running = False
        self.pygame.quit()
