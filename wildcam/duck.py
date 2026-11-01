"""Prepare a public time-lapse benchmark from "Duck pictures in wetlands" (LILA BC).

The dataset: ~676k images from fixed cameras over boreal wetlands in Finland (2025), each image
labelled at image level (empty, mallard/teal/goldeneye adult or duckling, unidentified waterbird,
crane, swan, mammal). About 98 % of the images are empty. There are no boxes and no capture times
in the metadata, but the JPEGs carry EXIF DateTimeOriginal.

The cameras mix two modes: a clock-aligned 10-minute time-lapse (each a 3-frame burst) and
motion-triggered bursts. Only the first frame of each time-lapse burst is kept here (see
``timelapse_frames``), so the benchmark matches the target setting: a fixed camera, a fixed
interval, rare and often small or distant animals, and hard negatives (water reflections, ripples,
wind in reeds, fog).

For each chosen camera folder this reads the EXIF time of every image with an HTTP range request,
keeps the time-lapse frames, takes the ``n`` consecutive ones containing the most animal frames,
crops the camera's info bar off the bottom, resizes to ``size`` and splits them in time (first
``train_frac`` for training). Labels are image level only:

    {"kind": "bird" | "mammal", "box": null}     animal present
    []                                           empty

Choosing the window with the most animals is a selection step: the positive rate of the result is
higher than that of the whole folder, and ``prepare`` reports both.
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

BASE = "https://storage.googleapis.com/public-datasets-lila/duck-pictures-in-wetlands/"
META = BASE + "duck_pictures_in_wetlands.json.zip"

# One folder per site, from both camera brands, with 2-7 % animal frames overall.
DEFAULT_FOLDERS = [
    "Dylanplotti/185HUNTI",
    "Lapinjarvi/Camera 2/100MEDIA",
    "Peikko plotti/100KDSP1",
    "Syrjanlusen rimpi/100HUNTI_BKUP/100HUNTI",
    "kuivaskorpi/100KDSP1",
    "hautrimpi/100HUNTI_BKUP/100HUNTI",
]


def _fetch(url: str, rng: str | None = None, retries: int = 4) -> bytes:
    req = urllib.request.Request(url.replace(" ", "%20"), headers={"Range": f"bytes={rng}"} if rng else {})
    for k in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if k == retries - 1:
                raise
    raise RuntimeError(url)


def _exif_time(name: str) -> str | None:
    from PIL import Image
    try:
        head = _fetch(BASE + name, "0-65535")
        raw = Image.open(io.BytesIO(head)).getexif().get_ifd(0x8769).get(0x9003)
        return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S").isoformat() if raw else None
    except Exception:
        return None


def load_metadata(cache: Path) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / "duck_pictures_in_wetlands.json"
    if not p.exists():
        zipfile.ZipFile(io.BytesIO(_fetch(META))).extractall(cache)
    return json.loads(p.read_text())


def _kind(species: set[str]) -> str | None:
    s = species - {"empty"}
    if not s:
        return None
    return "mammal" if s == {"mammal"} else "bird"


def capture_times(folder: str, names: list[str], cache: Path, workers: int = 32) -> dict[str, str | None]:
    p = cache / ("times_" + folder.replace("/", "_").replace(" ", "_") + ".json")
    if p.exists():
        return json.loads(p.read_text())
    with ThreadPoolExecutor(workers) as ex:
        times = dict(zip(names, ex.map(_exif_time, names)))
    p.write_text(json.dumps(times))
    return times


def timelapse_frames(items: list[tuple[datetime, str]], burst_gap: float = 60, period: int = 600,
                     tol: int = 5) -> list[tuple[datetime, str]]:
    """First frame of each scheduled time-lapse burst.

    The cameras shoot a short burst (3 frames, 1-2 s apart) every ``period`` seconds, aligned to
    the clock, and also shoot bursts when motion triggers them. Frames are grouped into bursts
    (gaps <= ``burst_gap`` s); a burst is a time-lapse burst when its start lies within ``tol`` s
    of the camera's dominant phase modulo ``period``. Motion-triggered bursts are dropped.
    """
    starts = [(t, n) for i, (t, n) in enumerate(items) if i == 0 or (t - items[i - 1][0]).total_seconds() > burst_gap]
    if not starts:
        return []
    phase = np.array([int(t.timestamp()) % period for t, _ in starts])
    counts = np.bincount(phase, minlength=period)
    window = np.array([counts.take(range(p - tol, p + tol + 1), mode="wrap").sum() for p in range(period)])
    p0 = int(np.argmax(window))
    dist = np.minimum((phase - p0) % period, (p0 - phase) % period)
    return [s for s, d in zip(starts, dist) if d <= tol]


def _best_window(items: list, positive: list[bool], n: int) -> list:
    if len(items) <= n:
        return items
    c = np.cumsum([0] + positive)
    s = int(np.argmax(c[n:] - c[:-n]))
    return items[s:s + n]


def prepare(out_root: str | Path, folders: list[str] | None = None, n: int = 864, train_frac: float = 0.5,
            size: tuple[int, int] = (480, 640), bar_frac: float = 0.045, burst_gap: float = 60,
            cache: str | Path = "data/duck_meta", workers: int = 16) -> list[dict]:
    cache = Path(cache)
    meta = load_metadata(cache)
    cat = {c["id"]: c["name"] for c in meta["categories"]}
    species = defaultdict(set)
    for a in meta["annotations"]:
        species[a["image_id"]].add(cat[a["category_id"]])
    stats = []
    for folder in folders or DEFAULT_FOLDERS:
        names = sorted(im["file_name"] for im in meta["images"] if im["file_name"].rsplit("/", 1)[0] == folder)
        times = capture_times(folder, names, cache)
        items = sorted((datetime.fromisoformat(t), nm) for nm, t in times.items() if t)
        tl = timelapse_frames(items, burst_gap)
        win = _best_window(tl, [_kind(species[nm]) is not None for _, nm in tl], n)
        cam_name = folder.split("/")[0].replace(" ", "_").lower()
        cut = int(len(win) * train_frac)
        for split, part in (("train", win[:cut]), ("test", win[cut:])):
            cam = Path(out_root) / split / cam_name
            cam.mkdir(parents=True, exist_ok=True)

            def work(item):
                t, nm = item
                name = f"{cam_name}_{t:%Y%m%d_%H%M%S}.jpg"
                dst = cam / name
                if not dst.exists():
                    img = cv2.imdecode(np.frombuffer(_fetch(BASE + nm), np.uint8), cv2.IMREAD_COLOR)
                    img = img[: int(img.shape[0] * (1 - bar_frac))]       # drop the date/temperature bar
                    cv2.imwrite(str(dst), cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA),
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                k = _kind(species[nm])
                return name, ([{"kind": k, "box": None, "species": sorted(species[nm] - {"empty"})}] if k else [])

            with ThreadPoolExecutor(workers) as ex:
                labels = dict(ex.map(work, part))
            (cam / "labels.json").write_text(json.dumps(labels))
            s = {"split": split, "camera": cam_name, "folder": folder, "images": len(part),
                 "animal": sum(bool(v) for v in labels.values()),
                 "folder_images": len(names), "folder_timelapse": len(tl),
                 "folder_timelapse_animal": sum(_kind(species[nm]) is not None for _, nm in tl),
                 "from": part[0][0].isoformat(), "to": part[-1][0].isoformat()}
            log.info("%s", s)
            stats.append(s)
    return stats
