"""Change map -> candidate regions. Small blobs near the horizon are grouped into herds."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Candidate:
    box: tuple[int, int, int, int]      # x0, y0, x1, y1
    zone: str                           # "band" (distant) | "ground" (near) | "md" (MegaDetector)
    area: int
    z_max: float
    z_mean: float
    n_blobs: int = 1
    prob: float = 0.0                   # verifier probability
    score: float = 0.0                  # after temporal support
    support: int = 0
    feats: np.ndarray | None = field(default=None, repr=False)

    @property
    def center(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2


def _union(boxes):
    b = np.array(boxes)
    return int(b[:, 0].min()), int(b[:, 1].min()), int(b[:, 2].max()), int(b[:, 3].max())


def _group(blobs: list[dict], eps: float) -> list[list[dict]]:
    """Single-linkage grouping of blob centres (union-find; herds are a handful of blobs)."""
    parent = list(range(len(blobs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(blobs)):
        for j in range(i + 1, len(blobs)):
            if np.hypot(*(np.subtract(blobs[i]["c"], blobs[j]["c"]))) <= eps:
                parent[find(i)] = find(j)
    groups: dict[int, list] = {}
    for i, b in enumerate(blobs):
        groups.setdefault(find(i), []).append(b)
    return list(groups.values())


def propose(z: np.ndarray, band: np.ndarray, ground: np.ndarray, cfg: dict) -> list[Candidate]:
    p = cfg["proposals"]
    binary = (z > p["z_thresh"]).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    band_blobs, out = [], []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        cx, cy = cents[i]
        zone = "band" if band[int(cy), int(cx)] else "ground" if ground[int(cy), int(cx)] else "sky"
        vals = z[labels == i]
        blob = {"box": (int(x), int(y), int(x + w), int(y + h)), "area": int(area), "c": (cx, cy),
                "z_max": float(vals.max()), "z_sum": float(vals.sum())}
        if zone == "band" and p["min_area_band"] <= area <= p["max_area_band"]:
            band_blobs.append(blob)
        elif zone == "ground" and p["min_area_ground"] <= area <= p["max_area_ground"]:
            out.append(Candidate(blob["box"], "ground", blob["area"], blob["z_max"], blob["z_sum"] / area))
        # sky: birds and clouds; animals of interest are never there
    for g in _group(band_blobs, p["herd_eps"]):
        area = sum(b["area"] for b in g)
        out.append(Candidate(_union([b["box"] for b in g]), "band", area, max(b["z_max"] for b in g),
                             sum(b["z_sum"] for b in g) / area, n_blobs=len(g)))
    return out
