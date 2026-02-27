import random
import time
import math
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Actions (8-direction movement)
# ----------------------------
ACTIONS = [
    (0, -1),    # 0  UP
    (0,  1),    # 1  DOWN
    (-1, 0),    # 2  LEFT
    (1,  0),    # 3  RIGHT
    (-1, -1),   # 4  UP-LEFT
    (1,  -1),   # 5  UP-RIGHT
    (-1,  1),   # 6  DOWN-LEFT
    (1,   1),   # 7  DOWN-RIGHT
]
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "UL", "UR", "DL", "DR"]
N_ACTIONS = len(ACTIONS)

def argmax_index(values):
    m = max(values)
    idxs = [i for i, v in enumerate(values) if v == m]
    return random.choice(idxs)

# ----------------------------
# 3x3 pooled into 4 overlapping 2x2 quadrants: UL, UR, DR, DL
# ----------------------------
QUADS = [
    [(-1, -1), (0, -1), (-1, 0), (0, 0)],  # UL
    [(0, -1), (1, -1), (0, 0), (1, 0)],    # UR
    [(0, 0), (1, 0), (0, 1), (1, 1)],      # DR
    [(-1, 0), (0, 0), (-1, 1), (0, 1)],    # DL
]

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def normalize(xs, eps=1e-9):
    s = sum(xs) + eps
    return [x / s for x in xs]

def bucketize(x, bins):
    for i, b in enumerate(bins):
        if x < b:
            return i
    return len(bins)

# ----------------------------
# Matplotlib live renderer
# ----------------------------
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

# ----------------------------
# Environment
# ----------------------------
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

        # --- base reward ---
        step_cost = 0.05
        tile = self.grid[y][x]  # '0' blue, '1' red
        base_reward = 1 if tile == "0" else -1
        r = base_reward - step_cost

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

# ----------------------------
# Random grid with red border
# ----------------------------
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

# ----------------------------
# Ant-like state: quadrant pooling + relative strengths + bucketing
# ----------------------------
def get_state(env):
    """
    Observation:
      - Pool local tiles (3x3 around agent) into 4 quadrant averages (UL,UR,DR,DL)
      - Pool local pheromones similarly
      - Convert each set into relative strengths summing to 1
      - Bucketize strengths -> discrete state for tabular Q-learning

    Tile encoding for pooling:
      blue ('0') = +1
      red  ('1') = -1
      out-of-bounds never happens (clamped), but treat as -2 if needed.
    """
    x0, y0 = env.pos

    tile_avgs = []
    pher_avgs = []

    for quad in QUADS:
        t_sum = 0.0
        p_sum = 0.0
        for dx, dy in quad:
            x, y = x0 + dx, y0 + dy
            if not (0 <= x < env.W and 0 <= y < env.H):
                t_val = -2.0
                p_val = 1.0
            else:
                t_val = 1.0 if env.grid[y][x] == "0" else -1.0
                p_val = env.P[y][x]
            t_sum += t_val
            p_sum += p_val

        tile_avgs.append(t_sum / 4.0)
        pher_avgs.append(p_sum / 4.0)

    tile_strength = softmax(tile_avgs)  # sums to 1

    # lower pheromone is better -> goodness then normalize
    pher_goodness = [1.0 / (1.0 + p) for p in pher_avgs]
    pher_strength = normalize(pher_goodness)  # sums to 1

    # Discretize into buckets
    bins = [0.1, 0.3, 0.5, 0.7]  # 5 buckets => 0..4
    tile_b = tuple(bucketize(v, bins) for v in tile_strength)
    pher_b = tuple(bucketize(v, bins) for v in pher_strength)

    return tile_b + pher_b  # 8 small integers

# ----------------------------
# Q-learning (live window visualization)
# ----------------------------
def q_learning(
    env,
    episodes=200,
    max_steps=200,
    alpha=0.2,
    gamma=0.95,
    eps=0.4,
    visualize=True,
    delay=0.02,
    visualize_every_episode=1,
    show_pheromone=True,
):
    Q = defaultdict(lambda: [0.0] * N_ACTIONS)

    renderer = None
    if visualize:
        renderer = MatplotlibRenderer(env, show_pheromone=show_pheromone, scale_pheromone=1.0)

    for ep in range(1, episodes + 1):
        env.reset()
        s = get_state(env)
        total_r = 0.0

        show = visualize and (ep % visualize_every_episode == 0)

        for t in range(1, max_steps + 1):
            exploring = (random.random() < eps)
            if exploring:
                a = random.randrange(N_ACTIONS)
            else:
                a = argmax_index(Q[s])

            _, r, done = env.step(a)
            total_r += r
            s2 = get_state(env)

            best_next = max(Q[s2])
            Q[s][a] += alpha * (r + gamma * best_next - Q[s][a])

            if show and renderer:
                renderer.update(
                    title=(
                        f"TRAIN  ep {ep}/{episodes}  t {t}/{max_steps}  "
                        f"a={ACTION_NAMES[a]} ({'explore' if exploring else 'exploit'})  "
                        f"r={r:.3f}  total={total_r:.2f}  eps={eps:.3f}"
                    )
                )
                time.sleep(delay)

            s = s2
            if done:
                break

        eps = max(0.01, eps * 0.995)

    if renderer:
        renderer.update(title="Training done.")
    return Q

# ----------------------------
# Greedy rollout (live window)
# ----------------------------
def greedy_run(env, Q, max_steps=200, delay=0.05, show_pheromone=True):
    env.reset()
    renderer = MatplotlibRenderer(env, show_pheromone=show_pheromone, scale_pheromone=1.0)

    total_r = 0.0
    for t in range(max_steps):
        s = get_state(env)
        a = argmax_index(Q[s])

        _, r, done = env.step(a)
        total_r += r

        renderer.update(
            title=f"GREEDY  t {t+1}/{max_steps}  a={ACTION_NAMES[a]}  r={r:.3f}  total={total_r:.2f}"
        )
        time.sleep(delay)

        if done:
            renderer.update(title=f"Reached endpoint in {t+1} steps. Total reward={total_r:.2f}")
            return total_r, True

    renderer.update(title=f"Max steps reached. Total reward={total_r:.2f}")
    return total_r, False

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    grid = random_grid(W=12, H=8, p_blue=0.7)

    # place start/goal away from red border
    start = (1, 1)
    goal = (10, 6)

    env = GridWorld(grid, start=start, goal=goal)

    Q = q_learning(
        env,
        episodes=200,
        max_steps=200,
        alpha=0.2,
        gamma=0.95,
        eps=0.4,
        visualize=True,
        delay=0.02,
        visualize_every_episode=1,  # set to 10 if too much
        show_pheromone=True,
    )

    greedy_run(env, Q, max_steps=200, delay=0.05, show_pheromone=True)

    # Keep window open after script ends
    plt.ioff()
    plt.show()