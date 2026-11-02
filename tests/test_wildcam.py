from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from wildcam import synth, temporal
from wildcam.background import BackgroundModel
from wildcam.config import load_config
from wildcam.evaluate import evaluate
from wildcam.frames import timestamp
from wildcam.horizon import band_masks, estimate_horizon
from wildcam.proposals import Candidate, propose


def test_horizon_found_on_synthetic_scene():
    static, hz = synth.render_static(3)
    est = estimate_horizon(static.clip(0, 255).astype(np.uint8))
    assert np.median(np.abs(est - hz)) < 3


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.uniform(60, 160, (120, 160)).astype(np.float32), (0, 0), 1.5)
    return np.repeat(base[..., None], 3, 2)


def _feed(model, frames, t0=datetime(2023, 7, 1, 12)):
    out = None
    for k, f in enumerate(frames):
        out = model.update(np.clip(f, 0, 255).astype(np.uint8), t0 + timedelta(days=k))
    return out


def test_background_ignores_shadow_but_finds_small_dark_spot():
    rng = np.random.default_rng(1)
    base = _scene()
    history = [base + rng.normal(0, 2, base.shape) for _ in range(8)]
    frame = base.copy()
    frame[:, :80] *= 0.6                      # broad cloud shadow over half the image
    frame[60:63, 120:124] -= 45               # a 3x4 px animal
    z, _ = _feed(BackgroundModel(buffer=8, min_frames=4), history + [frame])
    assert z[60:63, 120:124].max() > 4
    assert np.percentile(z[:, :80], 99) < 3   # the shadow itself does not fire


def test_previous_frame_mode_warms_up_after_one_frame():
    m = BackgroundModel(mode="previous")
    a = _scene().astype(np.uint8)
    assert m.update(a, datetime(2023, 1, 1))[0] is None
    assert m.update(a, datetime(2023, 1, 1, 0, 10))[0] is not None


def test_band_blobs_are_grouped_into_one_herd():
    cfg = load_config()
    hz = np.full(200, 50.0, np.float32)
    _, band, ground = band_masks(hz, (120, 200), 10)
    z = np.zeros((120, 200), np.float32)
    for x in (40, 52, 61, 70):                # four animals near the horizon
        z[54:56, x:x + 3] = 9
    z[100:110, 150:160] = 9                   # one larger change on the ground
    cands = propose(z, band, ground, cfg)
    herd = [c for c in cands if c.zone == "band"]
    assert len(herd) == 1 and herd[0].n_blobs == 4 and herd[0].box[0] == 40
    assert [c.zone for c in cands].count("ground") == 1


def test_temporal_support_rewards_persistent_candidates():
    def c(x, p):
        return Candidate((x, 10, x + 6, 14), "band", 10, 5, 5, prob=p, score=p)
    frames = [[c(100, 0.6)], [c(105, 0.6)], [c(110, 0.6), c(400, 0.6)], [], []]
    temporal.apply(frames, window=2, max_shift=10, bonus=0.15)
    assert frames[2][0].support == 2 and abs(frames[2][0].score - 0.9) < 1e-6
    assert frames[2][1].support == 0 and frames[2][1].score == 0.6


def test_timestamp_from_filename(tmp_path):
    p = tmp_path / "cam07_20230704_153000.jpg"
    p.write_bytes(b"")
    assert timestamp(p) == datetime(2023, 7, 4, 15, 30)


def test_evaluate_counts_false_positives_per_10k():
    labels = {"a.jpg": [{"kind": "herd", "box": [10, 10, 20, 15]}], "b.jpg": [], "c.jpg": []}
    res = [{"image": "a.jpg", "status": "ok", "score": 0.9, "candidates": [{"box": [11, 10, 19, 14], "score": 0.9}]},
           {"image": "b.jpg", "status": "ok", "score": 0.8, "candidates": [{"box": [0, 0, 5, 5], "score": 0.8}]},
           {"image": "c.jpg", "status": "ok", "score": 0.1, "candidates": []}]
    r = evaluate(res, labels, review=0.3, animal=0.7)
    assert r["at_animal_threshold"]["recall"] == 1.0
    assert r["at_animal_threshold"]["false_positives_per_10k_empty"] == 5000.0
    assert r["box_recall_at_review"]["herd"] == 1.0


def test_synthetic_camera_writes_labels(tmp_path):
    stats = synth.write_camera(tmp_path, "camX", scene_seed=5, seed=6, days=1, herds_per_day=4)
    assert stats["frames"] == 144 and stats["herd"] > 0
    assert (Path(tmp_path) / "camX" / "labels.json").exists()


def test_fit_mil_finds_positive_instance_in_bags():
    import numpy as np
    from wildcam.pipeline import fit_mil
    from wildcam.verifier import Verifier
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (300, 4)); y = np.zeros(300, int)            # negatives near 0
    bags = [np.vstack([rng.normal(0, 1, (5, 4)), rng.normal(3, 0.5, (1, 4))]) for _ in range(20)]
    v = fit_mil(Verifier(), X, y, bags)
    p_pos = v.model.predict_proba(rng.normal(3, 0.5, (50, 4)))[:, 1]
    p_neg = v.model.predict_proba(rng.normal(0, 1, (50, 4)))[:, 1]
    assert p_pos.mean() > 0.8 and p_neg.mean() < 0.5


def test_alignment_cancels_camera_shift():
    import numpy as np
    from datetime import datetime, timedelta
    from wildcam.background import BackgroundModel, estimate_shift, shift_image
    rng = np.random.default_rng(1)
    base = (rng.random((120, 160)) * 200).astype(np.float32)
    base = __import__("cv2").GaussianBlur(base, (0, 0), 2) * 3
    assert np.allclose(estimate_shift(base, shift_image(base, 5, -3)), (5, -3), atol=0.3)
    t0 = datetime(2025, 6, 1, 12)
    res = {}
    for align in (False, True):
        bgm = BackgroundModel(buffer=6, min_frames=3, blur=0, shadow_window=0, align=align)
        for k in range(6):
            img = np.repeat(base[..., None], 3, 2).clip(0, 255).astype(np.uint8)
            bgm.update(img, t0 + timedelta(minutes=10 * k))
        moved = np.repeat(shift_image(base, 6, 4)[..., None], 3, 2).clip(0, 255).astype(np.uint8)
        z, _ = bgm.update(moved, t0 + timedelta(minutes=70))
        res[align] = float((z > 3).mean())
    assert res[False] > 0.03 and res[True] < res[False] / 10, res


def test_idaho_timelapse_frames_and_labels():
    from wildcam.idaho import label, timelapse_frames
    ims = [{"id": str(i), "datetime": f"2016-01-01 10:{m:02d}:{s:02d}"}
           for i, (m, s) in enumerate([(0, 0), (5, 0), (7, 13), (7, 14), (10, 0), (20, 0), (33, 0)])]
    kept = [im["datetime"][-5:] for im in timelapse_frames(ims)]
    assert kept == ["00:00", "05:00", "10:00"]          # triggers, isolated 10:20, off-grid 10:33 dropped
    assert label({"empty"}) == [] and label({"foggy weather"}) == []
    assert label({"elk", "empty"})[0]["kind"] == "animal" and label({"human"})[0]["kind"] == "human"


def test_background_max_age_and_global_norm():
    from datetime import datetime, timedelta
    from wildcam.background import BackgroundModel
    rng = np.random.default_rng(0)
    day1 = [rng.normal(100, 2, (60, 80, 3)).clip(0, 255).astype(np.uint8) for _ in range(6)]
    bright = rng.normal(100, 25, (60, 80, 3)).clip(0, 255).astype(np.uint8)     # next day: new texture everywhere (snow)
    t0 = datetime(2016, 1, 1, 10)

    def changed(**kw):
        m = BackgroundModel(buffer=12, bucket_hours=3, min_frames=4, **kw)
        for i, im in enumerate(day1):
            m.update(im, t0 + timedelta(minutes=5 * i))
        z, _ = m.update(bright, t0 + timedelta(days=1))
        return None if z is None else float((z > 3).mean())

    stale = changed()
    assert stale > 0.1                           # stale background: much of the frame looks changed
    assert changed(max_age_hours=2) is None      # stale frames dropped -> warm-up instead of false alarms
    assert changed(global_norm=True) < stale / 5
