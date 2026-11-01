"""Temporal support: a real herd is usually present in neighbouring frames too, a few pixels away.

Trackers such as SORT assume continuous motion and do not suit 10-minute intervals; instead each
candidate gains a bonus for every neighbouring frame (within ``window``) that has a candidate in
the same zone within ``max_shift`` px per frame of time difference.
"""
from __future__ import annotations

import numpy as np


def apply(frames: list[list], window: int, max_shift: float, bonus: float, min_prob: float = 0.3) -> None:
    for i, cands in enumerate(frames):
        for c in cands:
            support = 0
            for j in range(max(0, i - window), min(len(frames), i + window + 1)):
                if j == i:
                    continue
                lim = max_shift * abs(i - j)
                if any(o.zone == c.zone and o.prob >= min_prob and
                       np.hypot(o.center[0] - c.center[0], o.center[1] - c.center[1]) <= lim for o in frames[j]):
                    support += 1
            c.support = support
            c.score = min(1.0, c.prob + bonus * support)
