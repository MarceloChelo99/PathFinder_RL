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

# 3x3 pooled into 4 overlapping 2x2 quadrants: UL, UR, DR, DL
QUADS = [
    [(-1, -1), (0, -1), (-1, 0), (0, 0)],  # UL
    [(0, -1), (1, -1), (0, 0), (1, 0)],    # UR
    [(0, 0), (1, 0), (0, 1), (1, 1)],      # DR
    [(-1, 0), (0, 0), (-1, 1), (0, 1)],    # DL
]
