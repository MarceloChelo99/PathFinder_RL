from core.environment import GridWorld, random_grid
from core.training import greedy_run, q_learning


if __name__ == "__main__":
    # Visualization controls
    TRAIN_VISUALIZE = True
    GREEDY_VISUALIZE = True

    # Larger map for richer navigation behavior.
    grid = random_grid(W=24, H=16, p_blue=0.72)

    # place start/goal away from the border walls
    start = (1, 1)
    goal = (22, 14)

    env = GridWorld(grid, start=start, goal=goal)

    Q = q_learning(
        env,
        episodes=220,
        max_steps=300,
        alpha=0.2,
        gamma=0.95,
        eps=0.4,
        visualize=TRAIN_VISUALIZE,
        delay=0.02,
        visualize_every_episode=1,
        show_pheromone=True,
    )

    greedy_run(
        env,
        Q,
        max_steps=300,
        delay=0.05,
        show_pheromone=True,
        visualize=GREEDY_VISUALIZE,
    )
