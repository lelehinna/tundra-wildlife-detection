"""Horizon estimation from a camera's long-term median image.

Distant animals on open tundra appear as a few pixels just below the skyline, so the proposal
stage uses a lower area threshold inside a band around it. Per column, the horizon is the row
with the strongest bright-to-dark vertical transition (sky above, ground below); a median filter
across columns removes columns fooled by rocks or clouds.
"""
from __future__ import annotations

import cv2
import numpy as np


def estimate_horizon(median_rgb: np.ndarray, smooth_columns: int = 61, search=(0.05, 0.85)) -> np.ndarray:
    g = cv2.GaussianBlur(cv2.cvtColor(median_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 2.0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=5)
    h = g.shape[0]
    lo, hi = int(search[0] * h), int(search[1] * h)
    rows = lo + np.argmin(gy[lo:hi], axis=0)           # most negative = brightness drops going down
    k = smooth_columns if smooth_columns % 2 else smooth_columns + 1
    pad = np.pad(rows, k // 2, mode="edge")
    return np.array([np.median(pad[i:i + k]) for i in range(len(rows))], np.float32)


def band_masks(horizon: np.ndarray, shape: tuple[int, int], band_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(sky, band, ground) boolean masks."""
    yy = np.arange(shape[0])[:, None]
    band = np.abs(yy - horizon[None]) <= band_px
    sky = (yy < horizon[None]) & ~band
    ground = (yy > horizon[None]) & ~band
    return sky, band, ground
