"""Grid world environment used for tabular Q-learning."""

import random

from core.constants import ACTIONS


class GridWorld:
    def __init__(self, grid, start, goal):
        """
        grid: list[str], chars '0' (free) and '1' (obstacle/wall)
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
        tx, ty = x + dx, y + dy

        hit_wall = not (0 <= tx < self.W and 0 <= ty < self.H)
        nx = max(0, min(self.W - 1, tx))
        ny = max(0, min(self.H - 1, ty))

        self.pos = (nx, ny)
        x, y = self.pos

        tile = self.grid[y][x]  # '0' free, '1' obstacle
        hit_obstacle = tile == "1"

        # Free tile is neutral; obstacle/wall is penalized.
        tile_reward = 0.0 if tile == "0" else -1.0
        r = tile_reward
        if hit_wall:
            r -= 1.0

        # --- pheromone dynamics in [0,1] ---
        decay = 0.97
        for yy in range(self.H):
            row = self.P[yy]
            for xx in range(self.W):
                row[xx] *= decay

        deposit_rate = 0.6
        self.P[y][x] += deposit_rate * (1.0 - self.P[y][x])

        pher_penalty = 1.0
        r -= pher_penalty * self.P[y][x]

        reached_goal = self.pos == self.goal
        done = reached_goal or hit_wall or hit_obstacle
        if reached_goal:
            r += 50

        info = {
            "hit_wall": hit_wall,
            "hit_obstacle": hit_obstacle,
            "reached_goal": reached_goal,
        }
        return self.pos, r, done, info


def random_grid(W=12, H=8, p_blue=0.7):
    """
    Red border, random interior:
      '0' = free cell
      '1' = obstacle/wall
    """
    grid = []
    for y in range(H):
        row = []
        for x in range(W):
            if x == 0 or x == W - 1 or y == 0 or y == H - 1:
                row.append("1")
            else:
                row.append("0" if random.random() < p_blue else "1")
        grid.append("".join(row))
    return grid
