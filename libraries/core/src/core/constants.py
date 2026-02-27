"""Shared constants for the path-finding RL domain."""
from enum import Enum

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

VISION_RADIUS = 2


def _quad_offsets(dx_sign, dy_sign, radius=VISION_RADIUS):
    """Offsets in a directional quadrant out to the configured vision radius."""
    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            if dx_sign < 0 and dx > 0:
                continue
            if dx_sign > 0 and dx < 0:
                continue
            if dy_sign < 0 and dy > 0:
                continue
            if dy_sign > 0 and dy < 0:
                continue
            offsets.append((dx, dy))
    return offsets


# 5x5 (radius=2) pooled into 4 directional quadrants: UL, UR, DR, DL
QUADS = [
    _quad_offsets(dx_sign=-1, dy_sign=-1),  # UL
    _quad_offsets(dx_sign=1, dy_sign=-1),   # UR
    _quad_offsets(dx_sign=1, dy_sign=1),    # DR
    _quad_offsets(dx_sign=-1, dy_sign=1),   # DL
]
