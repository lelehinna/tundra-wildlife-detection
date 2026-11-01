"""Time-of-day background model and normalised change maps.

For every frame the background is the per-pixel median of the last ``buffer`` frames taken in the
same time-of-day bucket, so lighting is comparable (cf. Jenkins et al. 2024, who build the
background from 12 prior images of the same period). Before differencing, the frame's low
frequencies are matched to the background's, which cancels global exposure changes and drifting
cloud shadows while keeping anything a few pixels wide: the frame/background ratio is median
filtered (``shadow_window``), so shadow edges are followed but small animals are not. The difference is expressed as a robust
z-score against the per-pixel spread (MAD) of the buffer, so naturally noisy pixels (grass, water)
need a larger change to fire than stable ones.

``mode="previous"`` replaces all of this with plain differencing against the previous frame; it is
kept as the baseline for ablations.
"""
from __future__ import annotations

from collections import defaultdict, deque

import cv2
import numpy as np


def to_gray(img: np.ndarray, blur: float) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), blur) if blur > 0 else g


def _masked_blur(x: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    m = mask.astype(np.float32)
    return cv2.GaussianBlur(x * m, (0, 0), sigma) / (cv2.GaussianBlur(m, (0, 0), sigma) + 1e-6)


def match_illumination(frame: np.ndarray, ref: np.ndarray, regions: list[np.ndarray] | None = None,
                       sigma: float = 25.0) -> np.ndarray:
    """Rescale ``frame`` so its low frequencies match ``ref``. With ``regions`` (e.g. sky and
    ground) each region is normalised on its own, so the bright sky does not bleed into the
    ground estimate along the horizon."""
    if not regions:
        return frame * ((cv2.GaussianBlur(ref, (0, 0), sigma) + 1) / (cv2.GaussianBlur(frame, (0, 0), sigma) + 1))
    out = frame.copy()
    for m in regions:
        gain = (_masked_blur(ref, m, sigma) + 1) / (_masked_blur(frame, m, sigma) + 1)
        out[m] = (frame * gain)[m]
    return out


def local_light_ratio(frame: np.ndarray, ref: np.ndarray, window: int) -> np.ndarray:
    """Per-pixel illumination ratio frame/ref, estimated with a median filter. Cloud shadows and
    exposure changes are wider than ``window`` and are captured; animals narrower than about
    half the window are treated as outliers and left in the residual."""
    r = np.clip(frame / (ref + 1.0), 0, 2.55)
    k = window if window % 2 else window + 1
    return cv2.medianBlur((r * 100).astype(np.uint8), k).astype(np.float32) / 100 + 1e-3


class BackgroundModel:
    def __init__(self, buffer: int = 12, bucket_hours: int = 3, min_frames: int = 4, blur: float = 1.0,
                 mode: str = "median", noise_floor: float = 2.0, shadow_window: int = 25):
        self.buffer, self.bucket_hours, self.min_frames = buffer, bucket_hours, min_frames
        self.shadow_window = shadow_window
        self.blur, self.mode, self.noise_floor = blur, mode, noise_floor
        self.buffers: dict[int, deque] = defaultdict(lambda: deque(maxlen=buffer))
        self.prev: np.ndarray | None = None
        self.regions: list[np.ndarray] | None = None

    def set_regions(self, *masks: np.ndarray) -> None:
        """Regions normalised separately for illumination, e.g. above / below the horizon."""
        self.regions = list(masks)

    @classmethod
    def from_config(cls, cfg: dict, mode: str = "median") -> "BackgroundModel":
        b = cfg["background"]
        return cls(b["buffer"], b["bucket_hours"], b["min_frames"], b["blur"], mode,
                   shadow_window=b.get("shadow_window", 25))

    def bucket(self, t) -> int:
        return t.hour // self.bucket_hours

    def update(self, img: np.ndarray, t) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Returns (z-score map, background gray image) for this frame, then adds it to the model.
        (None, None) while the bucket has fewer than ``min_frames`` frames."""
        g = to_gray(img, self.blur)
        if self.mode == "previous":
            out = (None, None)
            if self.prev is not None:
                d = np.abs(g - self.prev)
                out = (d / (np.median(d) * 1.4826 + self.noise_floor), self.prev)
            self.prev = g
            return out
        buf = self.buffers[self.bucket(t)]
        out = (None, None)
        if len(buf) >= self.min_frames:
            stack = np.stack(buf)
            bg = np.median(stack, axis=0)
            mad = np.median(np.abs(stack - bg), axis=0) * 1.4826
            if self.shadow_window:
                light = local_light_ratio(g, bg, self.shadow_window)
                out = (np.abs(g - bg * light) / (mad * light + self.noise_floor), bg)
            else:
                g_n = match_illumination(g, bg, self.regions)
                out = (np.abs(g_n - bg) / (mad + self.noise_floor), bg)
        buf.append(g)
        return out
