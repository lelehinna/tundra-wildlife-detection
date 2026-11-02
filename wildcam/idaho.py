"""Prepare a public time-lapse benchmark from "Idaho Camera Traps" (LILA BC).

The dataset: ~1.5M images from 276 Reconyx cameras of the Idaho Department of Fish and Game,
January-April 2016, mostly open sagebrush and snow-covered hills, many with a skyline and animals
(elk, deer, pronghorn) far from the camera. 76 locations ran a 5-minute time-lapse (marked "T" in
the info bar) alongside motion triggers ("M"). Labels are per sequence; every time-lapse frame is
its own one-image sequence, so for these frames the label is per image. There are no boxes.

Time-lapse frames are taken from the metadata alone: a frame on the 5-minute grid (``ts % 300 ==
0``) with another frame exactly 300 s before or after it. Motion-triggered frames carry arbitrary
seconds and are dropped. For each chosen location this takes the ``n`` consecutive time-lapse
frames containing the most animal frames, crops the info bars (top and bottom), resizes to
``size`` and splits them in time (first ``train_frac`` for training). Labels are image level:

    {"kind": "animal" | "human", "box": null, "species": [...]}     something present
    []                                                               empty (fog, snow on lens included)

Choosing the window with the most animals is a selection step; ``prepare`` reports both the
window's and the location's positive rate.
"""
from __future__ import annotations

import io
import json
import logging
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .duck import _best_window, _fetch

log = logging.getLogger(__name__)

BASE = "https://storage.googleapis.com/public-datasets-lila/idaho-camera-traps/"
IMAGES = BASE + "public/"
META = BASE + "idaho-camera-traps.json.zip"

# Time-lapse locations with the most animal frames, mostly elk and deer; 36 adds distant elk herds.
DEFAULT_LOCATIONS = ["16", "3", "44", "46", "28", "51", "12", "36"]

NOT_PRESENT = {"empty", "snow on lens", "foggy lens", "foggy weather", "malfunction", "misdirected",
               "lens obscured", "sun", "tilted", "vegetation obstruction", "unknown", "other"}
PEOPLE = {"human", "vehicle", "domestic dog", "cattle", "horse"}


def load_metadata(cache: Path) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / "idaho-camera-traps.json"
    if not p.exists():
        zipfile.ZipFile(io.BytesIO(_fetch(META))).extractall(cache)
    return json.loads(p.read_text())


def label(species: set[str]) -> list[dict]:
    s = species - NOT_PRESENT
    if not s:
        return []
    kind = "animal" if s - PEOPLE else "human"
    return [{"kind": kind, "box": None, "species": sorted(s)}]


def timelapse_frames(images: list[dict], period: int = 300) -> list[dict]:
    t = {}
    for im in images:
        t.setdefault(int(datetime.fromisoformat(im["datetime"]).timestamp()), im)
    keep = [im for ts, im in t.items() if ts % period == 0 and (ts - period in t or ts + period in t)]
    return sorted(keep, key=lambda im: im["datetime"])


def prepare(out_root: str | Path, locations: list[str] | None = None, n: int = 1152, train_frac: float = 0.5,
            size: tuple[int, int] = (480, 640), top_frac: float = 0.022, bottom_frac: float = 0.025,
            cache: str | Path = "data/idaho_meta", workers: int = 24) -> list[dict]:
    meta = load_metadata(Path(cache))
    cat = {c["id"]: c["name"] for c in meta["categories"]}
    species = defaultdict(set)
    for a in meta["annotations"]:
        species[a["image_id"]].add(cat[a["category_id"]])
    by_loc = defaultdict(list)
    for im in meta["images"]:
        by_loc[im["location"]].append(im)
    stats = []
    for loc in locations or DEFAULT_LOCATIONS:
        tl = timelapse_frames(by_loc[loc])
        win = _best_window(tl, [bool(label(species[im["id"]])) for im in tl], n)
        cam_name = f"loc{int(loc):04d}"
        cut = int(len(win) * train_frac)
        for split, part in (("train", win[:cut]), ("test", win[cut:])):
            cam = Path(out_root) / split / cam_name
            cam.mkdir(parents=True, exist_ok=True)

            def work(im):
                t = datetime.fromisoformat(im["datetime"])
                name = f"{cam_name}_{t:%Y%m%d_%H%M%S}.jpg"
                dst = cam / name
                if not dst.exists():
                    img = cv2.imdecode(np.frombuffer(_fetch(IMAGES + im["file_name"]), np.uint8), cv2.IMREAD_COLOR)
                    h = img.shape[0]
                    img = img[int(h * top_frac): int(h * (1 - bottom_frac))]      # drop both info bars
                    cv2.imwrite(str(dst), cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA),
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                return name, label(species[im["id"]])

            with ThreadPoolExecutor(workers) as ex:
                labels = dict(ex.map(work, part))
            (cam / "labels.json").write_text(json.dumps(labels))
            s = {"split": split, "camera": cam_name, "images": len(part),
                 "animal": sum(any(g["kind"] == "animal" for g in v) for v in labels.values()),
                 "human": sum(any(g["kind"] == "human" for g in v) for v in labels.values()),
                 "location_timelapse": len(tl),
                 "location_timelapse_animal": sum(bool(label(species[im["id"]])) for im in tl),
                 "from": part[0]["datetime"], "to": part[-1]["datetime"]}
            log.info("%s", s)
            stats.append(s)
    return stats
