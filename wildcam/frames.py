"""Frame discovery and timestamps."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

_TS = re.compile(r"(\d{8})[_-]?(\d{6})")
_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass
class Frame:
    path: Path
    time: datetime

    @property
    def name(self) -> str:
        return self.path.name


def timestamp(path: Path) -> datetime:
    """From the file name (..._YYYYmmdd_HHMMSS...), else EXIF DateTimeOriginal, else file mtime."""
    m = _TS.search(path.stem)
    if m:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    try:
        from PIL import Image
        exif = Image.open(path).getexif()
        raw = exif.get_ifd(0x8769).get(0x9003) or exif.get(0x0132)
        if raw:
            return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def list_frames(camera_dir: str | Path) -> list[Frame]:
    paths = [p for p in Path(camera_dir).iterdir() if p.suffix.lower() in _EXTS]
    return sorted((Frame(p, timestamp(p)) for p in paths), key=lambda f: f.time)


def list_cameras(root: str | Path) -> list[Path]:
    return sorted(p for p in Path(root).iterdir() if p.is_dir())


def read_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size and img.shape[:2] != tuple(size):
        img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return img
