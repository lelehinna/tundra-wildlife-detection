"""Synthetic tundra time-lapse cameras, for running and testing the pipeline without field data.

Each camera looks across open tundra at a fixed horizon. Frames are taken every 10 minutes around
the clock (arctic summer). The fixed scene varies with sun elevation, cloud cover, drifting cloud
shadows, sensor noise and the odd 1-2 px camera jitter. Rare events:

* "near": a single caribou-like animal in the foreground, 1-3 frames,
* "herd": 4-15 distant animals a few pixels in size and low in contrast, just below the horizon,
  drifting sideways for 3-8 consecutive frames,
* nuisances: birds crossing the sky (single frame), cloud shadows on the ground.

It is a smoke-test environment: real cameras add fog, rain on the lens, snow, heat shimmer,
auto-exposure jumps and JPEG artefacts, none of which are modelled here.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

H, W = 480, 640
INTERVAL_MIN = 10


def horizon_curve(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.arange(W)
    y = 150 + rng.uniform(-20, 20) + 10 * np.sin(x / rng.uniform(80, 160) + rng.uniform(0, 6)) \
        + 4 * np.sin(x / rng.uniform(20, 40))
    return y.astype(np.float32)


def render_static(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed scene as float32 RGB plus the horizon row per column."""
    rng = np.random.default_rng(seed)
    hz = horizon_curve(seed)
    yy = np.arange(H)[:, None].astype(np.float32)
    sky = (yy < hz[None])
    img = np.zeros((H, W, 3), np.float32)
    t = (yy / H)
    img[...] = np.stack([150 + 60 * t, 175 + 50 * t, 215 + 20 * t], -1) * np.ones((1, W, 1))  # sky gradient
    # Ground: tundra browns/greens, coarser texture towards the camera (bottom).
    noise = np.zeros((H, W), np.float32)
    for s, a in ((2, 10), (6, 14), (20, 18)):
        n = cv2.resize(rng.normal(0, 1, (H // s + 1, W // s + 1)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        noise += a * n
    depth = np.clip((yy - hz[None]) / (H - hz[None]), 0, 1)  # 0 at horizon, 1 at bottom
    base = np.stack([110 + 15 * depth, 105 + 10 * depth, 80 + 5 * depth], -1)
    ground = base + noise[..., None] * (0.4 + 0.6 * depth[..., None])
    for _ in range(60):  # rocks and shrubs
        x, y = rng.integers(0, W), rng.integers(int(hz.max()) + 10, H)
        r = int(2 + 10 * (y - hz.max()) / H)
        cv2.circle(ground, (int(x), int(y)), max(1, r), tuple(float(c) for c in rng.uniform(70, 130) * np.array([1.0, 0.97, 0.9])), -1)
    ridge = (np.abs(yy - hz[None]) < 3)[..., None]          # hazy distant ridge
    ground = np.where(ridge, ground * 0.85 + 25, ground)
    img = np.where(sky[..., None], img, ground)
    return img.astype(np.float32), hz


def sun_gain(t: datetime) -> float:
    h = t.hour + t.minute / 60
    return float(0.45 + 0.55 * max(0.0, np.sin(np.pi * (h - 2) / 22)))  # dim around 01:00, bright midday


class Weather:
    """Slowly drifting cloud field, shared by sky brightness and ground shadows."""

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        big = cv2.resize(rng.normal(0, 1, (12, 40)).astype(np.float32), (W * 4, H), interpolation=cv2.INTER_CUBIC)
        self.field = cv2.GaussianBlur(big, (0, 0), 12)
        self.cover = rng.uniform(0.2, 0.8)

    def at(self, k: int) -> np.ndarray:
        off = (k * 9) % (self.field.shape[1] - W)
        f = self.field[:, off:off + W]
        return np.clip((f - (1 - 2 * self.cover)) * 1.5, 0, 1)  # 1 = thick cloud


def _draw_caribou(img, x, y, scale, rng):
    c = np.array([95, 82, 70]) * rng.uniform(0.8, 1.15)
    bw, bh = int(14 * scale), int(7 * scale)
    cv2.ellipse(img, (int(x), int(y)), (bw, bh), 0, 0, 360, tuple(float(v) for v in c), -1)
    d = 1 if rng.random() < 0.5 else -1
    cv2.ellipse(img, (int(x + d * bw), int(y - bh)), (max(1, int(4 * scale)), max(1, int(3 * scale))), 0, 0, 360,
                tuple(float(v) for v in c * 0.9), -1)
    for lx in (-0.7, -0.4, 0.4, 0.7):
        cv2.line(img, (int(x + lx * bw), int(y + bh * 0.6)), (int(x + lx * bw), int(y + bh + 9 * scale)),
                 tuple(float(v) for v in c * 0.8), max(1, int(2 * scale)))
    cv2.ellipse(img, (int(x), int(y + bh + 9 * scale)), (bw, max(1, int(2 * scale))), 0, 0, 360, (60, 60, 50), -1)
    return (int(x - bw - 5 * scale), int(y - 2 * bh), int(x + bw + 5 * scale), int(y + bh + 10 * scale))


def _draw_herd(img, cx, hz, n, spread, contrast, rng):
    boxes = []
    for _ in range(n):
        x = cx + rng.normal(0, spread)
        if not 2 <= x < W - 2:
            continue
        y = hz[int(x)] + rng.uniform(2, 9)
        a, b = rng.integers(2, 4), rng.integers(1, 3)
        patch = img[int(y) - 3:int(y) + 4, int(x) - 4:int(x) + 5]
        m = np.zeros(patch.shape[:2], np.float32)
        cv2.ellipse(m, (4, 3), (int(a), int(b)), 0, 0, 360, 1.0, -1)
        patch -= contrast * m[..., None]
        boxes.append((int(x) - a - 1, int(y) - b - 1, int(x) + a + 1, int(y) + b + 1))
    if not boxes:
        return None
    b = np.array(boxes)
    return (int(b[:, 0].min()), int(b[:, 1].min()), int(b[:, 2].max()), int(b[:, 3].max()))


def write_camera(root: str | Path, cam: str, scene_seed: int, seed: int, days: int = 4, start: datetime | None = None,
                 near_per_day: float = 2.0, herds_per_day: float = 3.0, birds_per_day: float = 6.0,
                 jitter_prob: float = 0.02) -> dict:
    """Write root/<cam>/<cam>_YYYYmmdd_HHMMSS.jpg and root/<cam>/labels.json."""
    rng = np.random.default_rng(seed)
    out = Path(root) / cam
    out.mkdir(parents=True, exist_ok=True)
    static, hz = render_static(scene_seed)          # the camera's fixed view
    weather = Weather(seed + 1)                      # this period's weather and events
    n = days * 24 * 60 // INTERVAL_MIN
    start = start or datetime(2023, 7, 1)
    events: dict[int, list] = {}
    for _ in range(rng.poisson(near_per_day * days)):
        k0 = int(rng.integers(0, n))
        x, y = rng.uniform(60, W - 60), rng.uniform(hz.max() + 80, H - 40)
        for j in range(int(rng.integers(1, 4))):
            events.setdefault(k0 + j, []).append(("near", x + j * rng.uniform(-30, 30), y, int(rng.integers(1 << 30))))
    for _ in range(rng.poisson(herds_per_day * days)):
        k0 = int(rng.integers(0, n))
        cx, v = rng.uniform(40, W - 40), rng.uniform(-6, 6)
        size, spread, contrast = int(rng.integers(4, 16)), rng.uniform(6, 25), rng.uniform(14, 40)
        s = int(rng.integers(1 << 30))
        for j in range(int(rng.integers(3, 9))):
            events.setdefault(k0 + j, []).append(("herd", cx + v * j, size, spread, contrast, s + j))
    for _ in range(rng.poisson(birds_per_day * days)):
        events.setdefault(int(rng.integers(0, n)), []).append(("bird", rng.uniform(0, W), rng.uniform(20, hz.min() - 5)))

    labels = {}
    for k in range(n):
        t = start + timedelta(minutes=INTERVAL_MIN * k)
        img = static.copy()
        boxes = []
        for ev in events.get(k, []):
            r = np.random.default_rng(ev[-1] if ev[0] != "bird" else k)
            if ev[0] == "near":
                scale = 0.6 + 1.6 * (ev[2] - hz.max()) / (H - hz.max())
                boxes.append({"kind": "near", "box": list(_draw_caribou(img, ev[1], ev[2], scale, r))})
            elif ev[0] == "herd":
                b = _draw_herd(img, ev[1], hz, ev[2], ev[3], ev[4], r)
                if b:
                    boxes.append({"kind": "herd", "box": list(b)})
            else:  # bird: not an animal of interest here, an expected false-positive source
                cv2.ellipse(img, (int(ev[1]), int(ev[2])), (3, 1), 0, 0, 360, (40, 40, 45), -1)
        cloud = weather.at(k)
        yy = np.arange(H)[:, None]
        sky = yy < hz[None]
        light = sun_gain(t) * np.where(sky, 1 + 0.25 * cloud, 1 - 0.35 * cloud)  # brighter clouds, darker shadows
        img = img * light[..., None]
        if rng.random() < jitter_prob:
            M = np.float32([[1, 0, rng.uniform(-2, 2)], [0, 1, rng.uniform(-2, 2)]])
            img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        img = np.clip(img + rng.normal(0, 2.5, img.shape), 0, 255).astype(np.uint8)
        name = f"{cam}_{t:%Y%m%d_%H%M%S}.jpg"
        cv2.imwrite(str(out / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        labels[name] = boxes
    (out / "labels.json").write_text(json.dumps(labels))
    pos = sum(bool(v) for v in labels.values())
    return {"camera": cam, "frames": n, "frames_with_animals": pos,
            "near": sum(b["kind"] == "near" for v in labels.values() for b in v),
            "herd": sum(b["kind"] == "herd" for v in labels.values() for b in v)}


def write_dataset(root: str | Path, cameras: int = 2, days: int = 4, seed: int = 0) -> list[dict]:
    """Same cameras, later dates for the test split: root/train/<cam> and root/test/<cam>."""
    stats = []
    for c in range(cameras):
        cam = f"cam{c:02d}"
        stats.append({"split": "train", **write_camera(Path(root) / "train", cam, seed + 10 * c, seed + 10 * c + 1, days)})
        stats.append({"split": "test", **write_camera(Path(root) / "test", cam, seed + 10 * c, seed + 10 * c + 2, days,
                                                       start=datetime(2023, 7, 1) + timedelta(days=days))})
    return stats
