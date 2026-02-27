"""Matplotlib renderer for interactive simulation display."""

import matplotlib.pyplot as plt
import numpy as np


class MatplotlibRenderer:
    """
    Live window renderer:
      - Blue tiles (0) vs Red tiles (1)
      - Agent (C) black, Goal (E) green
      - Optional pheromone overlay (darkens visited areas)
    """

    def __init__(self, env, show_pheromone=True, scale_pheromone=1.0):
        self.env = env
        self.show_pheromone = show_pheromone
        self.scale_pheromone = scale_pheromone

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.im = self.ax.imshow(self._frame(), interpolation="nearest")
        self.text = self.ax.text(0.01, 1.01, "", transform=self.ax.transAxes)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _frame(self):
        H, W = self.env.H, self.env.W

        base = np.zeros((H, W), dtype=float)
        for y in range(H):
            for x in range(W):
                base[y, x] = 0.0 if self.env.grid[y][x] == "0" else 1.0

        img = np.zeros((H, W, 3), dtype=float)
        img[base == 0.0] = np.array([0.2, 0.5, 1.0])  # blue tiles
        img[base == 1.0] = np.array([1.0, 0.3, 0.3])  # red tiles

        if self.show_pheromone and hasattr(self.env, "P"):
            P = np.array(self.env.P, dtype=float)
            pnorm = np.clip(P / self.scale_pheromone, 0.0, 1.0)
            img = img * (1.0 - 0.6 * pnorm[..., None])

        gx, gy = self.env.goal
        img[gy, gx] = np.array([0.2, 1.0, 0.2])  # goal

        ax, ay = self.env.pos
        img[ay, ax] = np.array([0.0, 0.0, 0.0])  # agent

        return img

    def update(self, title=""):
        self.im.set_data(self._frame())
        self.text.set_text(title)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
