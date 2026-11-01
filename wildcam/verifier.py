"""Second stage: is this changed region an animal?

Features per candidate: CNN embedding of the frame crop, embedding of the same crop of the
background (so the model sees what changed, not just what is there), and shape statistics
(zone, area, number of blobs, change strength). A balanced logistic regression is enough because
the first stage has already removed nearly all of the empty images; most of the work is telling
animals from cloud-shadow edges, birds and jitter artefacts.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .proposals import Candidate


def square_crop(img: np.ndarray, box, pad: int, size: int) -> np.ndarray:
    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0, 16) / 2 + pad
    a0, a1 = int(max(0, cx - half)), int(min(w, cx + half))
    b0, b1 = int(max(0, cy - half)), int(min(h, cy + half))
    return cv2.resize(img[b0:b1, a0:a1], (size, size), interpolation=cv2.INTER_LINEAR)


def describe(cands: list[Candidate], gray: np.ndarray, bg: np.ndarray, embedder, cfg: dict) -> None:
    """Attach a feature vector to each candidate (in place)."""
    if not cands:
        return
    size, pad = cfg["verifier"]["crop"], cfg["proposals"]["pad"]
    fc = np.stack([square_crop(gray, c.box, pad, size) for c in cands]).clip(0, 255).astype(np.uint8)
    bc = np.stack([square_crop(bg, c.box, pad, size) for c in cands]).clip(0, 255).astype(np.uint8)
    ef, eb = embedder(fc), embedder(bc)
    for c, f, b in zip(cands, ef, eb):
        shape = [c.zone == "band", c.zone == "ground", c.zone == "md", np.log1p(c.area), np.log1p(c.n_blobs),
                 np.log1p(c.z_max), np.log1p(c.z_mean), (c.box[2] - c.box[0]) / max(1, c.box[3] - c.box[1])]
        c.feats = np.concatenate([f, f - b, np.asarray(shape, np.float32)])


class Verifier:
    def __init__(self):
        self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced"))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Verifier":
        self.model.fit(X, y)
        return self

    def predict(self, cands: list[Candidate]) -> None:
        if cands:
            p = self.model.predict_proba(np.stack([c.feats for c in cands]))[:, 1]
            for c, v in zip(cands, p):
                c.prob = c.score = float(v)

    def save(self, path) -> None:
        Path(path).write_bytes(pickle.dumps(self.model))

    @classmethod
    def load(cls, path) -> "Verifier":
        v = cls()
        v.model = pickle.loads(Path(path).read_bytes())
        return v
