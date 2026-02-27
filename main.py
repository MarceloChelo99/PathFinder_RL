import matplotlib.pyplot as plt

from core.environment import GridWorld, random_grid
from core.training import greedy_run, q_learning


if __name__ == "__main__":
    # Visualization controls
    TRAIN_VISUALIZE = True
    GREEDY_VISUALIZE = True

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
        visualize=TRAIN_VISUALIZE,
        delay=0.02,
        visualize_every_episode=1,
        show_pheromone=True,
    )

    greedy_run(
        env,
        Q,
        max_steps=200,
        delay=0.05,
        show_pheromone=True,
        visualize=GREEDY_VISUALIZE,
    )

    # Keep window open after script ends
    plt.ioff()
    plt.show()
