"""Training and evaluation loops for the tabular RL agent."""

import random
import time
from collections import defaultdict

from core.constants import ACTION_NAMES, N_ACTIONS
from core.renderer import MatplotlibRenderer
from core.state import get_state
from core.utils import argmax_index


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
            exploring = random.random() < eps
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


def greedy_run(env, Q, max_steps=200, delay=0.05, show_pheromone=True, visualize=True):
    env.reset()
    renderer = None
    if visualize:
        renderer = MatplotlibRenderer(env, show_pheromone=show_pheromone, scale_pheromone=1.0)

    total_r = 0.0
    for t in range(max_steps):
        s = get_state(env)
        a = argmax_index(Q[s])

        _, r, done = env.step(a)
        total_r += r

        if renderer:
            renderer.update(
                title=f"GREEDY  t {t+1}/{max_steps}  a={ACTION_NAMES[a]}  r={r:.3f}  total={total_r:.2f}"
            )
            time.sleep(delay)

        if done:
            if renderer:
                renderer.update(title=f"Reached endpoint in {t+1} steps. Total reward={total_r:.2f}")
            return total_r, True

    if renderer:
        renderer.update(title=f"Max steps reached. Total reward={total_r:.2f}")
    return total_r, False
