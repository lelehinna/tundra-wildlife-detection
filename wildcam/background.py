"""Time-of-day background model and normalised change maps.

For every frame the background is the per-pixel median of the last ``buffer`` frames taken in the
same time-of-day bucket, so lighting is comparable (cf. Jenkins et al. 2024, who build the
background from 12 prior images of the same period). Before differencing, the frame's low
frequencies are matched to the background's, which cancels global exposure changes and drifting
cloud shadows while keeping anything a few pixels wide: the frame/background ratio is median
filtered (``shadow_window``), so shadow edges are followed but small animals are not. The difference is expressed as a robust
z-score against the per-pixel spread (MAD) of the buffer, so naturally noisy pixels (grass, water)
need a larger change to fire than stable ones.

With ``align=True`` small camera movements (a mount shaken by wind, a camera nudged during a
visit) are compensated: each frame's translation against the bucket's background is estimated
with phase correlation, the background and MAD are shifted into the frame's coordinates (so
candidate boxes stay in image coordinates), and the frame is stored in the bucket's coordinates.
Frames that moved more than ``align_max_shift`` px give no change map and are not added.

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


def estimate_shift(ref: np.ndarray, img: np.ndarray, min_response: float = 0.05) -> tuple[float, float]:
    """Translation (dx, dy) with img(x, y) ~ ref(x - dx, y - dy); (0, 0) when the match is weak
    (e.g. dark, low-texture night frames)."""
    win = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ref.astype(np.float32), img.astype(np.float32), win)
    return (dx, dy) if resp >= min_response else (0.0, 0.0)


def shift_image(x: np.ndarray, dx: float, dy: float) -> np.ndarray:
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(x, m, x.shape[::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def border_mask(shape, dx: float, dy: float) -> np.ndarray:
    """False where a shift by (dx, dy) brought in pixels from outside the image."""
    h, w = shape
    m = np.ones(shape, bool)
    ix, iy = int(np.ceil(abs(dx))), int(np.ceil(abs(dy)))
    if ix:
        if dx > 0: m[:, :ix] = False
        else: m[:, w - ix:] = False
    if iy:
        if dy > 0: m[:iy] = False
        else: m[h - iy:] = False
    return m


class BackgroundModel:
    def __init__(self, buffer: int = 12, bucket_hours: int = 3, min_frames: int = 4, blur: float = 1.0,
                 mode: str = "median", noise_floor: float = 2.0, shadow_window: int = 25,
                 align: bool = False, align_max_shift: float = 80, align_min_response: float = 0.05,
                 max_age_hours: float | None = None, global_norm: bool = False):
        self.max_age_hours, self.global_norm = max_age_hours, global_norm
        self.buffer, self.bucket_hours, self.min_frames = buffer, bucket_hours, min_frames
        self.shadow_window = shadow_window
        self.align, self.align_max_shift, self.align_min_response = align, align_max_shift, align_min_response
        self.last_shift = (0.0, 0.0)
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
                   shadow_window=b.get("shadow_window", 25), align=b.get("align", False),
                   align_max_shift=b.get("align_max_shift", 80), align_min_response=b.get("align_min_response", 0.05),
                   max_age_hours=b.get("max_age_hours"), global_norm=b.get("global_norm", False))

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
        tb = self.buffers[self.bucket(t)]            # (time, gray) pairs
        if self.max_age_hours is not None:           # drop frames too old to describe the current scene
            while tb and (t - tb[0][0]).total_seconds() > self.max_age_hours * 3600:
                tb.popleft()
        buf = [x for _, x in tb]
        out = (None, None)
        dx = dy = 0.0
        if self.align and buf:
            ref = np.median(np.stack(buf), axis=0)
            dx, dy = estimate_shift(ref, g, self.align_min_response)
            if np.hypot(dx, dy) > self.align_max_shift:     # camera re-pointed: no usable background
                self.last_shift = (dx, dy)
                return out
        self.last_shift = (dx, dy)
        if len(buf) >= self.min_frames:
            stack = np.stack(buf)
            bg = np.median(stack, axis=0)
            mad = np.median(np.abs(stack - bg), axis=0) * 1.4826
            if dx or dy:
                bg, mad = shift_image(bg, dx, dy), shift_image(mad, dx, dy)
            if self.shadow_window:
                light = local_light_ratio(g, bg, self.shadow_window)
                z = np.abs(g - bg * light) / (mad * light + self.noise_floor)
            else:
                g_n = match_illumination(g, bg, self.regions)
                z = np.abs(g_n - bg) / (mad + self.noise_floor)
            if self.global_norm:                     # whole-scene change (snow, IR switch, light): rescale
                z = z / max(1.0, float(np.median(z)) / 0.6745)
            if dx or dy:
                z[~border_mask(z.shape, dx, dy)] = 0
            out = (z, bg)
        tb.append((t, shift_image(g, -dx, -dy) if (dx or dy) else g))
        return out
