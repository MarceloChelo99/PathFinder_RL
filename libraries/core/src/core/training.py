"""Training and evaluation loops for the tabular RL agent."""

import random
import time
from collections import defaultdict

from core.constants import ACTION_NAMES, N_ACTIONS
from core.renderer import PygameRenderer
from core.state import QUAD_NAMES, get_state, get_state_debug
from core.utils import argmax_index


def _termination_reason(info):
    if info.get("reached_goal"):
        return "goal"
    if info.get("hit_wall"):
        return "wall"
    if info.get("hit_obstacle"):
        return "obstacle"
    return "max_steps"


def _format_q_values(q_values):
    return "Q(s): " + " ".join(f"{ACTION_NAMES[i]}={q_values[i]:+.3f}" for i in range(N_ACTIONS))


def _format_quad_means(debug):
    tile_parts = [f"{name}:{value:.3f}" for name, value in zip(QUAD_NAMES, debug["tile_strength"])]
    pher_parts = [f"{name}:{value:.3f}" for name, value in zip(QUAD_NAMES, debug["pher_strength"])]
    return (
        "norm-tile=" + ", ".join(tile_parts),
        "norm-pher=" + ", ".join(pher_parts),
    )


def _debug_lines(renderer, q_values, debug):
    lines = [_format_q_values(q_values)]
    if renderer and renderer.show_quad_means:
        lines.extend(_format_quad_means(debug))
    return lines


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
        renderer = PygameRenderer(env, show_pheromone=show_pheromone, scale_pheromone=1.0)

    for ep in range(1, episodes + 1):
        env.reset()
        s = get_state(env)
        total_r = 0.0

        show = visualize and (ep % visualize_every_episode == 0)

        for t in range(1, max_steps + 1):
            q_s = Q[s]
            exploring = random.random() < eps
            if exploring:
                a = random.randrange(N_ACTIONS)
            else:
                a = argmax_index(q_s)

            _, r, done, info = env.step(a)
            total_r += r
            debug2 = get_state_debug(env)
            s2 = debug2["state"]

            best_next = 0.0 if done else max(Q[s2])
            Q[s][a] += alpha * (r + gamma * best_next - Q[s][a])

            if show and renderer:
                keep_running = renderer.update(
                    title=(
                        f"TRAIN ep {ep}/{episodes} t {t}/{max_steps} "
                        f"a={ACTION_NAMES[a]} ({'explore' if exploring else 'exploit'}) "
                        f"r={r:.3f} total={total_r:.2f} eps={eps:.3f}"
                    ),
                    subtitle=f"state={s2}  pos={env.pos}  goal={env.goal}",
                    detail_lines=_debug_lines(renderer, q_s, debug2),
                )
                if not keep_running:
                    renderer.close()
                    return Q
                time.sleep(delay)
            s = s2
            if not done:
                pass
                # if show and renderer:
                #     keep_running = renderer.update(
                #         title=(
                #             f"TRAIN ep {ep}/{episodes} ended by {_termination_reason(info)} at step {t}; "
                #             f"episode_total={total_r:.2f}"
                #         ),
                #         subtitle=f"terminal_state={s2}  terminal_pos={env.pos}",
                #         detail_lines=_debug_lines(renderer, Q[s], debug2),
                #     )
                #     if not keep_running:
                #         renderer.close()
                #         return Q
                #     time.sleep(delay)

        eps = max(0.01, eps * 0.995)

    if renderer:
        renderer.update(title="Training done.", subtitle="")
    return Q


def greedy_run(env, Q, max_steps=200, delay=0.05, show_pheromone=True, visualize=True):
    env.reset()
    renderer = None
    if visualize:
        renderer = PygameRenderer(env, show_pheromone=show_pheromone, scale_pheromone=1.0)

    total_r = 0.0
    for t in range(max_steps):
        s = get_state(env)
        q_s = Q[s]
        a = argmax_index(q_s)

        _, r, done, info = env.step(a)
        total_r += r
        debug2 = get_state_debug(env)
        s2 = debug2["state"]

        if renderer:
            keep_running = renderer.update(
                title=f"GREEDY t {t+1}/{max_steps} a={ACTION_NAMES[a]} r={r:.3f} total={total_r:.2f}",
                subtitle=f"state={s2}  pos={env.pos}  goal={env.goal}",
                detail_lines=_debug_lines(renderer, q_s, debug2),
            )
            if not keep_running:
                renderer.close()
                return total_r, False
            time.sleep(delay)

        if done:
            reason = _termination_reason(info)
            if renderer:
                renderer.update(
                    title=f"Rollout ended by {reason} in {t+1} steps. Total reward={total_r:.2f}",
                    subtitle=f"terminal_state={s2}  terminal_pos={env.pos}",
                    detail_lines=_debug_lines(renderer, Q[s2], debug2),
                )
            return total_r, info.get("reached_goal", False)

    if renderer:
        renderer.update(title=f"Max steps reached. Total reward={total_r:.2f}", subtitle="")
    return total_r, False
