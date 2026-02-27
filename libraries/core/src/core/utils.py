"""Utility functions used across environment and training."""

import math
import random


def argmax_index(values):
    m = max(values)
    idxs = [i for i, v in enumerate(values) if v == m]
    return random.choice(idxs)


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
