"""Command line interface.

    python -m wildcam.cli synth  data/synth                       # 2 cameras x 4 days, train and test
    python -m wildcam.cli fit    data/synth/train --out models/v1   # train the verifier
    python -m wildcam.cli run    data/synth/test  --model models/v1 --out runs/v1
    python -m wildcam.cli eval   runs/v1 data/synth/test
    python -m wildcam.cli duck   data/duck                         # public time-lapse benchmark (LILA)
    python -m wildcam.cli idaho  data/idaho                        # 5-min time-lapse, open terrain (LILA)
    python -m wildcam.cli review runs/v1 data/synth/test --out to_label/   # active learning export
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np


def _proc(a, verifier=None):
    from .config import load_config
    from .embed import CropEmbedder
    from .pipeline import CameraProcessor
    cfg = load_config(a.config)
    md = None
    if cfg["megadetector"]["enabled"]:
        from .megadetector import TiledMegaDetector
        m = cfg["megadetector"]
        md = TiledMegaDetector(m["model"], m["conf"], m["tile"], m["overlap"])
    return cfg, CameraProcessor(cfg, CropEmbedder(weights=a.weights), verifier, md, a.background)


def _synth(a):
    from .synth import write_dataset
    for s in write_dataset(a.root, a.cameras, a.days, a.seed):
        print(s)


def _cct(a):
    from .cct import prepare
    for s in prepare(a.root, a.locations.split(","), a.n, cache=a.cache):
        print(s)


def _duck(a):
    from .duck import prepare
    folders = a.folders.split(";") if a.folders else None
    for s in prepare(a.root, folders, a.n, cache=a.cache):
        print(s)


def _idaho(a):
    from .idaho import prepare
    for s in prepare(a.root, a.locations.split(",") if a.locations else None, a.n, cache=a.cache):
        print(s)


def _fit(a):
    from .frames import list_cameras
    from .pipeline import fit_mil, load_labels, mil_bags, training_set
    from .verifier import Verifier
    cfg, proc = _proc(a)
    Xs, ys, bags = [], [], []
    for cam in list_cameras(a.root):
        t = time.time()
        res, labels = proc.propose_camera(cam), load_labels(cam)
        X, y = training_set(res, labels)
        b = mil_bags(res, labels) if a.mil else []
        print(f"{cam.name}: {len(y)} proposals, {int(y.sum())} on animals, {len(b)} image-level bags "
              f"({time.time() - t:.0f}s)")
        Xs.append(X); ys.append(y); bags += b
    X, y = np.concatenate(Xs), np.concatenate(ys)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    v = fit_mil(Verifier(), X, y, bags) if a.mil else Verifier().fit(X, y)
    v.save(out / "verifier.pkl")
    (out / "fit.json").write_text(json.dumps({"proposals": int(len(y)), "positives": int(y.sum()), "bags": len(bags),
                                              "mil": a.mil, "background": a.background}, indent=2))
    print(f"verifier trained on {len(y)} proposals ({int(y.sum())} positive, {len(bags)} bags) -> {out}")


def _run(a):
    from .frames import list_cameras
    from .verifier import Verifier
    verifier = Verifier.load(Path(a.model) / "verifier.pkl") if a.model else None
    cfg, proc = _proc(a, verifier)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for cam in list_cameras(a.root):
        t = time.time()
        res = proc.run(cam)
        with open(out / f"{cam.name}.jsonl", "w") as f:
            for r in res:
                f.write(json.dumps(r.summary()) + "\n")
        tri = {k: sum(r.triage == k for r in res) for k in ("animal", "review", "empty")}
        print(f"{cam.name}: {len(res)} frames in {time.time() - t:.0f}s, triage {tri}")


def _eval(a):
    from .config import load_config
    from .evaluate import evaluate
    from .frames import list_cameras
    from .pipeline import load_labels
    cfg = load_config(a.config)
    results, labels = [], {}
    for cam in list_cameras(a.root):
        labels.update(load_labels(cam))
        p = Path(a.run) / f"{cam.name}.jsonl"
        results += [json.loads(l) for l in open(p)]
    if a.common_with:   # compare runs on the same images (warm-up frames differ between background modes)
        ok = set()
        for p in Path(a.common_with).glob("*.jsonl"):
            ok |= {r["image"] for r in map(json.loads, open(p)) if r["status"] == "ok"}
        results = [r for r in results if r["image"] in ok]
    rep = evaluate(results, labels, cfg["triage"]["review"], cfg["triage"]["animal"])
    (Path(a.run) / (a.save_as or "eval.json")).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


def _review(a):
    """Export the most uncertain candidates (score closest to 0.5) as crops for labelling."""
    from .config import load_config
    from .verifier import square_crop
    cfg = load_config(a.config)
    rows = []
    for p in sorted(Path(a.run).glob("*.jsonl")):
        for l in open(p):
            r = json.loads(l)
            rows += [(abs(c["score"] - 0.5), p.stem, r["image"], c) for c in r["candidates"]]
    rows.sort(key=lambda t: t[0])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    index = []
    for k, (_, cam, image, c) in enumerate(rows[:a.n]):
        img = cv2.imread(str(Path(a.root) / cam / image))
        h, w = cfg["image_size"]
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        name = f"{k:04d}_{Path(image).stem}.png"
        cv2.imwrite(str(out / name), square_crop(img, c["box"], 24, 128))
        index.append({"crop": name, "camera": cam, "image": image, **c})
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{len(index)} crops -> {out}")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser("wildcam")
    sp = ap.add_subparsers(required=True)

    def common(s):
        s.add_argument("--config"); s.add_argument("--weights", default="timm")
        s.add_argument("--background", default="median", choices=["median", "previous"])
    s = sp.add_parser("synth"); s.add_argument("root"); s.add_argument("--cameras", type=int, default=2)
    s.add_argument("--days", type=int, default=4); s.add_argument("--seed", type=int, default=0); s.set_defaults(fn=_synth)
    s = sp.add_parser("cct"); s.add_argument("root"); s.add_argument("--locations", default="46,43,64,51,18,33")
    s.add_argument("--n", type=int, default=600); s.add_argument("--cache", default="data/cct_meta"); s.set_defaults(fn=_cct)
    s = sp.add_parser("duck"); s.add_argument("root"); s.add_argument("--folders", help="';'-separated folders")
    s.add_argument("--n", type=int, default=864); s.add_argument("--cache", default="data/duck_meta"); s.set_defaults(fn=_duck)
    s = sp.add_parser("idaho"); s.add_argument("root"); s.add_argument("--locations", help="comma-separated location ids")
    s.add_argument("--n", type=int, default=1152); s.add_argument("--cache", default="data/idaho_meta"); s.set_defaults(fn=_idaho)
    s = sp.add_parser("fit"); s.add_argument("root"); s.add_argument("--out", required=True)
    s.add_argument("--mil", action="store_true", help="also learn from image-level labels (no boxes)"); common(s); s.set_defaults(fn=_fit)
    s = sp.add_parser("run"); s.add_argument("root"); s.add_argument("--model"); s.add_argument("--out", required=True)
    common(s); s.set_defaults(fn=_run)
    s = sp.add_parser("eval"); s.add_argument("run"); s.add_argument("root"); s.add_argument("--config")
    s.add_argument("--common-with", help="only images that are past warm-up in this other run")
    s.add_argument("--save-as", help="file name for the report inside the run directory"); s.set_defaults(fn=_eval)
    s = sp.add_parser("review"); s.add_argument("run"); s.add_argument("root"); s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=200); s.add_argument("--config"); s.set_defaults(fn=_review)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
