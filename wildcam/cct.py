"""Prepare a public benchmark from Caltech Camera Traps (CCT, LILA BC).

CCT: 243,100 images from 140 fixed camera locations in the south-western US, with species labels
for every image (about half are empty) and human-drawn boxes for ~63k of them. Every image carries
its location and capture time, so each location can be treated like one fixed camera and the
background model can be built from that camera's own history.

For each chosen location this takes the window of ``n`` consecutive images (by capture time) with
the fewest "non-empty but unboxed" images, downloads them resized to ``size``, and splits them in
time: the first ``train_frac`` for training, the rest for testing. Labels:

    {"kind": "animal", "box": [x0, y0, x1, y1]}   annotated animal
    {"kind": "animal", "box": null}               animal present, no box (image-level label only)
    []                                            empty image

Licence: CDLA-Permissive 1.0. Cite Beery et al., "Recognition in Terra Incognita", ECCV 2018.
Note that MegaDetector was trained on data that includes CCT, so it is not a fair baseline here.
"""
from __future__ import annotations

import io
import json
import logging
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

META = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/labels/"
IMAGES = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/agentmorris/lila-wildlife/caltech-unzipped/cct_images/"


def _fetch(url: str, retries: int = 4) -> bytes:
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if k == retries - 1:
                raise
    raise RuntimeError(url)


def load_metadata(cache: Path) -> tuple[dict, dict]:
    cache.mkdir(parents=True, exist_ok=True)
    images_json = cache / "caltech_images_20210113.json"
    if not images_json.exists():
        zipfile.ZipFile(io.BytesIO(_fetch(META + "caltech_camera_traps.json.zip"))).extractall(cache)
    bbox_json = cache / "caltech_bboxes_20200316.json"
    if not bbox_json.exists():
        bbox_json.write_bytes(_fetch(META + "caltech_bboxes_20200316.json"))
    return json.loads(images_json.read_text()), json.loads(bbox_json.read_text())


def _index(meta: dict, bboxes: dict):
    cat = {c["id"]: c["name"] for c in meta["categories"]}
    species = defaultdict(set)
    for a in meta["annotations"]:
        species[a["image_id"]].add(cat[a["category_id"]])
    bcat = {c["id"]: c["name"] for c in bboxes["categories"]}
    boxes = defaultdict(list)
    for a in bboxes["annotations"]:
        if "bbox" in a and bcat[a["category_id"]] != "empty":
            boxes[a["image_id"]].append(a["bbox"])
    boxed = {im["id"] for im in bboxes["images"]}
    by_loc = defaultdict(list)
    for im in meta["images"]:
        s = species[im["id"]]
        kind = "empty" if s == {"empty"} else "boxed" if im["id"] in boxed and boxes[im["id"]] else "unboxed"
        try:
            t = datetime.strptime(im["date_captured"], "%Y-%m-%d %H:%M:%S")
        except ValueError:      # a few records carry malformed dates; they cannot be ordered
            continue
        by_loc[im["location"]].append((t, im, kind))
    for v in by_loc.values():
        v.sort(key=lambda t: (t[0], t[1].get("frame_num", 0)))
    return by_loc, boxes


def _best_window(items: list, n: int) -> list:
    if len(items) <= n:
        return items
    bad = np.cumsum([0] + [k == "unboxed" for _, _, k in items])
    starts = np.arange(len(items) - n + 1)
    s = int(starts[np.argmin(bad[starts + n] - bad[starts])])
    return items[s:s + n]


def prepare(out_root: str | Path, locations: list[str], n: int = 600, train_frac: float = 0.5,
            size: tuple[int, int] = (480, 640), cache: str | Path = "data/cct_meta", workers: int = 8) -> list[dict]:
    meta, bboxes = load_metadata(Path(cache))
    by_loc, boxes = _index(meta, bboxes)
    stats = []
    for loc in locations:
        win = _best_window(by_loc[loc], n)
        cut = int(len(win) * train_frac)
        for split, part in (("train", win[:cut]), ("test", win[cut:])):
            cam = Path(out_root) / split / f"loc{loc}"
            cam.mkdir(parents=True, exist_ok=True)
            labels = {}

            def work(item):
                t, im, kind = item
                name = f"loc{loc}_{t:%Y%m%d_%H%M%S}_{im['id'][:8]}.jpg"
                sx, sy = size[1] / im["width"], size[0] / im["height"]
                if kind == "empty":
                    lab = []
                elif kind == "boxed":
                    lab = [{"kind": "animal", "box": [round(x * sx), round(y * sy), round((x + w) * sx), round((y + h) * sy)]}
                           for x, y, w, h in boxes[im["id"]]]
                else:
                    lab = [{"kind": "animal", "box": None}]
                dst = cam / name
                if not dst.exists():
                    img = cv2.imdecode(np.frombuffer(_fetch(IMAGES + im["file_name"]), np.uint8), cv2.IMREAD_COLOR)
                    cv2.imwrite(str(dst), cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA),
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                return name, lab

            with ThreadPoolExecutor(workers) as ex:
                for name, lab in ex.map(work, part):
                    labels[name] = lab
            (cam / "labels.json").write_text(json.dumps(labels))
            s = {"split": split, "camera": cam.name, "images": len(part),
                 "empty": sum(not v for v in labels.values()),
                 "boxed": sum(bool(v) and v[0]["box"] is not None for v in labels.values()),
                 "unboxed": sum(bool(v) and v[0]["box"] is None for v in labels.values()),
                 "from": part[0][0].isoformat(), "to": part[-1][0].isoformat()}
            log.info("%s", s)
            stats.append(s)
    return stats
