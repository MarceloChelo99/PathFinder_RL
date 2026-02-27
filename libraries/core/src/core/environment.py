"""Grid world environment used for tabular Q-learning."""

import random

from core.constants import ACTIONS


class GridWorld:
    def __init__(self, grid, start, goal):
        """
        grid: list[str], chars '0' (blue) and '1' (red)
        start, goal: (x,y)
        """
        self.grid = grid
        self.H = len(grid)
        self.W = len(grid[0])
        self.start = start
        self.goal = goal
        self.reset()

    def reset(self):
        self.pos = self.start
        # Pheromone map in [0,1]
        self.P = [[0.0 for _ in range(self.W)] for _ in range(self.H)]
        return self.pos

    def step(self, action_idx):
        dx, dy = ACTIONS[action_idx]
        x, y = self.pos
        nx, ny = x + dx, y + dy

        # Clamp to bounds; edges are red anyway
        nx = max(0, min(self.W - 1, nx))
        ny = max(0, min(self.H - 1, ny))

        self.pos = (nx, ny)
        x, y = self.pos

        # Blue tiles are neutral (0 reward), red tiles are penalized.
        tile = self.grid[y][x]  # '0' blue, '1' red
        tile_reward = 0.0 if tile == "0" else -1.0
        r = tile_reward

        # --- pheromone dynamics in [0,1] ---
        decay = 0.97
        for yy in range(self.H):
            row = self.P[yy]
            for xx in range(self.W):
                row[xx] *= decay

        # diminishing deposit, bounded to 1
        deposit_rate = 0.6
        self.P[y][x] += deposit_rate * (1.0 - self.P[y][x])

        # penalize high-pheromone cells (discourage loops)
        pher_penalty = 1.0
        r -= pher_penalty * self.P[y][x]

        done = (self.pos == self.goal)
        if done:
            r += 50  # terminal bonus (optional)

        return self.pos, r, done


def random_grid(W=12, H=8, p_blue=0.7):
    """
    Red border, random interior:
      '0' = blue (good)
      '1' = red  (bad)
    """
    grid = []
    for y in range(H):
        row = []
        for x in range(W):
            if x == 0 or x == W - 1 or y == 0 or y == H - 1:
                row.append("1")  # red border
            else:
                row.append("0" if random.random() < p_blue else "1")
        grid.append("".join(row))
    return grid
