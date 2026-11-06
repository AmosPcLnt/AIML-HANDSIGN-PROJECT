"""Run from the project folder:  python -m pytest"""
import numpy as np

from hand_features import (NUM_FEATURES, _palm_local, augment_points,
                           normalize_points, points_to_features)


def fake_hand(seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(50, 400, (21, 3)).astype(np.float32)
    raw[9] = raw[0] + [30, -80, 5]   # make sure the palm length is non-trivial
    return raw


def test_translation_invariant():
    raw = fake_hand()
    a = normalize_points(raw)
    b = normalize_points(raw + np.array([120.0, -40.0, 0.0]))
    assert np.allclose(a, b, atol=1e-4)


def test_scale_invariant():
    raw = fake_hand()
    assert np.allclose(normalize_points(raw), normalize_points(raw * 2.5), atol=1e-4)


def test_left_hand_matches_mirrored_right_hand():
    raw = fake_hand()
    mirrored = raw * np.array([-1.0, 1.0, 1.0])
    assert np.allclose(normalize_points(raw, "Right"),
                       normalize_points(mirrored, "Left"), atol=1e-4)


def test_degenerate_hand_returns_none():
    assert normalize_points(np.zeros((21, 3))) is None


def test_feature_shapes_single_and_batch():
    pts = normalize_points(fake_hand())
    assert points_to_features(pts).shape == (NUM_FEATURES,)
    batch = np.stack([pts, pts, pts])
    out = points_to_features(batch)
    assert out.shape == (3, NUM_FEATURES)
    assert np.allclose(out[0], points_to_features(pts))
    assert np.isfinite(out).all()


def test_augmentation_keeps_wrist_and_shapes():
    pts = np.stack([normalize_points(fake_hand(s)) for s in range(5)])
    aug, src = augment_points(pts, np.random.default_rng(1), copies=3)
    assert aug.shape == (15, 21, 3)
    assert list(src) == [0, 1, 2, 3, 4] * 3
    assert np.allclose(aug[:, 0, :], 0.0, atol=1e-6)


def test_palm_frame_is_rotation_invariant():
    rng = np.random.default_rng(3)
    pts = normalize_points(fake_hand())
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1            # proper rotation only
    rotated = pts @ q.T
    assert np.allclose(_palm_local(pts), _palm_local(rotated), atol=1e-4)
