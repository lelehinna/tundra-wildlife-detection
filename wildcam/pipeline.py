"""Per-camera processing: proposals for every frame, verification, temporal support, triage."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import temporal
from .background import BackgroundModel, to_gray
from .frames import Frame, list_frames, read_rgb
from .horizon import band_masks, estimate_horizon
from .proposals import Candidate, propose
from .verifier import Verifier, describe

log = logging.getLogger(__name__)


@dataclass
class FrameResult:
    frame: Frame
    status: str                                  # "ok" | "warmup"
    candidates: list[Candidate] = field(default_factory=list)
    score: float = 0.0
    triage: str = "empty"

    def summary(self) -> dict:
        return {"image": self.frame.name, "time": self.frame.time.isoformat(), "status": self.status,
                "score": round(self.score, 4), "triage": self.triage,
                "candidates": [{"box": list(c.box), "zone": c.zone, "area": c.area, "n_blobs": c.n_blobs,
                                "prob": round(c.prob, 4), "score": round(c.score, 4), "support": c.support}
                               | ({"source": c.source} if c.source else {}) for c in self.candidates]}


def camera_horizon(frames: list[Frame], size, cfg: dict, sample: int = 60) -> np.ndarray:
    idx = np.linspace(0, len(frames) - 1, min(sample, len(frames))).astype(int)
    med = np.median(np.stack([read_rgb(frames[i].path, size) for i in idx]), axis=0).astype(np.uint8)
    return estimate_horizon(med, cfg["horizon"]["smooth_columns"])


class CameraProcessor:
    def __init__(self, cfg: dict, embedder, verifier: Verifier | None = None, megadetector=None,
                 background_mode: str = "median"):
        self.cfg, self.embedder, self.verifier, self.md = cfg, embedder, verifier, megadetector
        self.background_mode = background_mode

    def propose_camera(self, camera_dir: str | Path) -> list[FrameResult]:
        """First stage for one camera: candidates with features, no verification yet."""
        size = tuple(self.cfg["image_size"])
        frames = list_frames(camera_dir)
        if not frames:
            return []
        dual = self.background_mode == "dual"
        bgm = BackgroundModel.from_config(self.cfg, "median" if dual else self.background_mode)
        prev = BackgroundModel.from_config(self.cfg, "previous") if dual else None
        if self.cfg["horizon"].get("enabled", True):
            hz = camera_horizon(frames, size, self.cfg)
            _, band, ground = band_masks(hz, size, self.cfg["horizon"]["band_px"])
            above = np.arange(size[0])[:, None] < hz[None]
            bgm.set_regions(above, ~above)
            if prev is not None:
                prev.set_regions(above, ~above)
        else:   # close-range camera traps: no skyline, the whole frame is "ground"
            band, ground = np.zeros(size, bool), np.ones(size, bool)
        out = []
        for f in frames:
            img = read_rgb(f.path, size)
            z, bg = bgm.update(img, f.time)
            zp = prev.update(img, f.time)[0] if prev is not None else None
            if z is None:
                out.append(FrameResult(f, "warmup"))
                continue
            cands = propose(z, band, ground, self.cfg)
            if dual:
                cands = combine(cands, propose(zp, band, ground, self.cfg) if zp is not None else [], z, zp)
            describe(cands, to_gray(img, 0), bg, self.embedder, self.cfg)
            if self.md is not None:
                cands += self.md(img)
            out.append(FrameResult(f, "ok", cands))
        return out

    def finish(self, results: list[FrameResult]) -> list[FrameResult]:
        """Verification, temporal support and triage."""
        t, tr = self.cfg["temporal"], self.cfg["triage"]
        for r in results:
            own = [c for c in r.candidates if c.zone != "md"]
            if self.verifier is not None:
                self.verifier.predict(own)
            else:   # no verifier yet: rank by change strength only
                for c in own:
                    c.prob = c.score = float(1 - np.exp(-c.z_max / 10))
        temporal.apply([r.candidates for r in results], t["window"], t["max_shift"], t["bonus"])
        for r in results:
            r.score = max((c.score for c in r.candidates), default=0.0)
            r.triage = "animal" if r.score >= tr["animal"] else "review" if r.score >= tr["review"] else "empty"
        return results

    def run(self, camera_dir) -> list[FrameResult]:
        return self.finish(self.propose_camera(camera_dir))


def _box_stats(z: np.ndarray | None, box) -> tuple[float, float]:
    if z is None:
        return 0.0, 0.0
    x0, y0, x1, y1 = box
    v = z[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
    return (float(v.max()), float(v.mean())) if v.size else (0.0, 0.0)


def combine(med: list[Candidate], prv: list[Candidate], z_med: np.ndarray, z_prev: np.ndarray | None) -> list[Candidate]:
    """Dual background: all candidates of the bucket median (static herds survive), plus those of
    the previous frame that no median candidate overlaps. Each candidate carries the change
    strength under both backgrounds so the verifier can learn how to weigh them."""
    out = []
    for c in med:
        c.source = "both" if any(box_hit(c.box, o.box, 0) for o in prv) else "median"
        out.append(c)
    for c in prv:
        if not any(box_hit(c.box, o.box, 0) for o in med):
            c.source = "previous"
            out.append(c)
    for c in out:
        mz, mm = _box_stats(z_med, c.box)
        pz, pm = _box_stats(z_prev, c.box)
        c.extra = np.log1p([mz, mm, pz, pm]).tolist() + [float(c.source == "both")]
    return out


def load_labels(camera_dir: str | Path) -> dict[str, list[dict]]:
    p = Path(camera_dir) / "labels.json"
    return json.loads(p.read_text()) if p.exists() else {}


def box_hit(box, gt, slack: int = 6) -> bool:
    """Candidate box overlaps the (slightly enlarged) ground-truth box."""
    return not (box[2] < gt[0] - slack or box[0] > gt[2] + slack or box[3] < gt[1] - slack or box[1] > gt[3] + slack)


def training_set(results: list[FrameResult], labels: dict) -> tuple[np.ndarray, np.ndarray]:
    """Label every proposal by overlap with annotated boxes: the same data a person would produce
    by labelling harvested candidates."""
    X, y = [], []
    for r in results:
        labs = labels.get(r.frame.name, [])
        if any(g["box"] is None for g in labs):
            continue    # animal present but not boxed: proposals cannot be labelled
        gts = [g["box"] for g in labs]
        for c in r.candidates:
            if c.feats is not None:
                X.append(c.feats)
                y.append(int(any(box_hit(c.box, g) for g in gts)))
    return np.stack(X), np.array(y)


def mil_bags(results: list[FrameResult], labels: dict) -> list[np.ndarray]:
    """Feature bags of frames with an animal but no box (image-level labels only)."""
    bags = []
    for r in results:
        labs = labels.get(r.frame.name, [])
        if labs and all(g["box"] is None for g in labs):
            feats = [c.feats for c in r.candidates if c.feats is not None]
            if feats:
                bags.append(np.stack(feats))
    return bags


def fit_mil(verifier: Verifier, X: np.ndarray, y: np.ndarray, bags: list[np.ndarray], rounds: int = 3) -> Verifier:
    """Multiple-instance training for image-level labels (MI-max style).

    Proposals of empty frames (and box-labelled proposals) are instance labels. A positive frame
    without boxes is a bag: at least one of its proposals is the animal, if any proposal covers it
    at all. Round 0 treats every proposal of a bag as positive; each later round keeps only the
    proposal the current model scores highest in each bag, and refits.
    """
    pos = np.concatenate(bags) if bags else np.zeros((0, X.shape[1]), X.dtype)
    for _ in range(rounds):
        verifier.fit(np.concatenate([X, pos]), np.concatenate([y, np.ones(len(pos), int)]))
        pos = np.stack([b[np.argmax(verifier.model.predict_proba(b)[:, 1])] for b in bags]) if bags else pos
    return verifier
