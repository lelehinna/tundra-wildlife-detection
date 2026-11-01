"""Optional near-animal detector: MegaDetector with tiled inference.

MegaDetector finds animals, people and vehicles in camera-trap images and is the standard first
pass for clearly visible animals; its authors note it is not meant for extremely small, distant
objects, which is what the background/horizon branch covers. Requires `pip install megadetector`
(weights download from GitHub on first use). Not exercised by the tests.
"""
from __future__ import annotations

import numpy as np

from .proposals import Candidate


class TiledMegaDetector:
    def __init__(self, model: str = "MDV5A", conf: float = 0.2, tile: int = 640, overlap: float = 0.15):
        from megadetector.detection.run_detector import load_detector
        self.det = load_detector(model)
        self.conf, self.tile, self.step = conf, tile, int(tile * (1 - overlap))

    def __call__(self, img: np.ndarray) -> list[Candidate]:
        from PIL import Image
        h, w = img.shape[:2]
        out = []
        ys = list(range(0, max(1, h - self.tile) + 1, self.step)) or [0]
        xs = list(range(0, max(1, w - self.tile) + 1, self.step)) or [0]
        for y0 in ys:
            for x0 in xs:
                tile = img[y0:y0 + self.tile, x0:x0 + self.tile]
                res = self.det.generate_detections_one_image(Image.fromarray(tile), "tile", detection_threshold=self.conf)
                th, tw = tile.shape[:2]
                for d in res.get("detections", []):
                    if d["category"] != "1":   # 1 = animal
                        continue
                    bx, by, bw, bh = d["bbox"]
                    box = (int(x0 + bx * tw), int(y0 + by * th), int(x0 + (bx + bw) * tw), int(y0 + (by + bh) * th))
                    out.append(Candidate(box, "md", (box[2] - box[0]) * (box[3] - box[1]), 0.0, 0.0,
                                         prob=float(d["conf"]), score=float(d["conf"])))
        return out
