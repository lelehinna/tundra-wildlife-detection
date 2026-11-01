"""Evaluation at the level that matters for ecologists: which images to keep.

* image level: average precision, recall and false positives per 10k empty images at the triage
  thresholds (with 0.05 % positives, a 1 % false-positive rate would bury the real detections);
* by target type: recall on frames with distant herds vs. frames with near animals;
* box level: share of annotated animals/herds covered by a candidate scoring >= the review threshold.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from .pipeline import box_hit


def evaluate(results: list[dict], labels: dict, review: float, animal: float) -> dict:
    scores, y, kinds = [], [], []
    box_total: dict[str, int] = {}
    box_hit_n: dict[str, int] = {}
    for r in results:
        if r["status"] != "ok":
            continue
        gts = labels.get(r["image"], [])
        scores.append(r["score"])
        y.append(int(bool(gts)))
        kinds.append({g["kind"] for g in gts})
        for g in gts:
            if g["box"] is None:     # image-level label only
                continue
            box_total[g["kind"]] = box_total.get(g["kind"], 0) + 1
            if any(c["score"] >= review and box_hit(c["box"], g["box"]) for c in r["candidates"]):
                box_hit_n[g["kind"]] = box_hit_n.get(g["kind"], 0) + 1
    s, y = np.array(scores), np.array(y)
    neg = max(1, int((y == 0).sum()))

    def at(th):
        pred = s >= th
        rec = {k: round(float(pred[[k in ks for ks in kinds]].mean()), 3) for k in sorted(set().union(*kinds))}
        return {"recall": round(float(pred[y == 1].mean()), 3) if y.any() else None,
                "recall_by_kind": rec,
                "false_positives_per_10k_empty": round(float(pred[y == 0].sum()) / neg * 1e4, 1),
                "images_flagged": int(pred.sum())}

    return {"images": int(len(y)), "positives": int(y.sum()),
            "average_precision": round(float(average_precision_score(y, s)), 3) if y.any() else None,
            "at_review_threshold": at(review), "at_animal_threshold": at(animal),
            "box_recall_at_review": {k: round(box_hit_n.get(k, 0) / v, 3) for k, v in box_total.items()}}
