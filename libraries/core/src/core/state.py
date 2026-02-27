"""State featurization for local observations."""

from core.constants import QUADS
from core.utils import bucketize, normalize, softmax


QUAD_NAMES = ("UL", "UR", "DR", "DL")


def get_state_debug(env):
    """Return state plus intermediate pooled/normalized features for debugging."""
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
                t_val = 0.0 if env.grid[y][x] == "0" else -1.0
                p_val = env.P[y][x]
            t_sum += t_val
            p_sum += p_val

        n_tiles = float(len(quad))
        tile_avgs.append(t_sum / n_tiles)
        pher_avgs.append(p_sum / n_tiles)

    tile_strength = tile_avgs #softmax(tile_avgs)  # sums to 1

    # lower pheromone is better -> goodness then normalize
    pher_goodness = [1.0 / (1.0 + p) for p in pher_avgs]
    pher_strength = normalize(pher_goodness)  # sums to 1

    # Discretize into buckets
    bins = [0.2, 0.4, 0.6, 0.8]  # 5 buckets => 0..4
    tile_b = tuple(tile_strength) #tuple(bucketize(v, bins) for v in tile_strength)
    pher_b = tuple(bucketize(v, bins) for v in pher_strength)

    return {
        "state": tile_b + pher_b,
        "tile_avgs": tuple(tile_avgs),
        "pher_avgs": tuple(pher_avgs),
        "tile_strength": tuple(tile_strength),
        "pher_strength": tuple(pher_strength),
        "tile_buckets": tile_b,
        "pher_buckets": pher_b,
    }


def get_state(env):
    """
    Observation:
      - Pool local tiles (5x5 around agent, radius=2) into 4 quadrant averages (UL,UR,DR,DL)
      - Pool local pheromones similarly
      - Convert each set into relative strengths summing to 1
      - Bucketize strengths -> discrete state for tabular Q-learning

    Tile encoding for pooling:
      blue ('0') = +1
      red  ('1') = -1
      out-of-bounds never happens (clamped), but treat as -2 if needed.
    """
    return get_state_debug(env)["state"]
