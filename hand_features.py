"""Shared hand-tracking + feature extraction.

Two layers, deliberately separate:

1. CANONICAL POINTS (21 x 3): what gets stored in the CSV. Wrist at the
   origin, scaled by palm size, left hands mirrored to look like right hands.
   Because this layer is simple and stable, you can change the engineered
   features below without re-collecting any data.

2. ENGINEERED FEATURES: canonical points plus fingertip distances, joint
   angles, finger-spread angles and thumb-to-finger-base distances. These are
   computed at train and inference time from the canonical points.

Bump FEATURE_VERSION whenever either layer changes; train.py stores it in the
model bundle and recognize.py refuses to load a mismatched model.

Note: full rotation normalization is intentionally NOT applied, because hand
orientation distinguishes letters (e.g. K vs P, G vs Q). Rotation robustness
comes from augmentation in training instead (see augment_points).
"""
from dataclasses import dataclass
from itertools import combinations

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

FEATURE_VERSION = 3

NUM_LANDMARKS = 21
NUM_POINT_VALUES = NUM_LANDMARKS * 3
POINT_COLUMNS = [f"p{i}" for i in range(NUM_POINT_VALUES)]

_EPS = 1e-8
_CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
           [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]   # wrist -> fingertip
_TIPS = [4, 8, 12, 16, 20]
_FINGER_DIRS = [(1, 4), (5, 8), (9, 12), (13, 16), (17, 20)]  # (base, tip)
_THUMB_REFS = [5, 6, 9, 10, 13, 14, 17, 18]  # helps M / N / T / S / A
_THUMB_DIPS = [7, 11, 15, 19]


@dataclass
class Detection:
    landmarks: object        # MediaPipe landmarks, for drawing
    points: np.ndarray       # (21, 3) canonical points (stored in the CSV)
    features: np.ndarray     # engineered feature vector (fed to the model)
    hand: str                # "Left" or "Right" as reported by MediaPipe
    score: float             # handedness / detection confidence


def create_hands():
    # One hand is enough for fingerspelling; it also keeps features unambiguous.
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def normalize_points(raw, hand="Right"):
    """Raw landmark coordinates (pixels) -> canonical (21, 3) points.

    - translation: wrist is the origin
    - scale: divided by wrist-to-middle-knuckle (palm) length, which is stable
      regardless of how far the fingers are extended
    - handedness: left hands are mirrored to look like right hands
    Returns None for a degenerate hand.
    """
    pts = np.array(raw, dtype=np.float32)
    pts -= pts[0]
    palm = float(np.linalg.norm(pts[9]))
    if palm < _EPS:
        return None
    pts /= palm
    if hand == "Left":
        pts[:, 0] *= -1
    return pts


def _angle(a, b, c):
    """Angle at b between segments b->a and b->c (radians, last axis = xyz)."""
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (
        np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + _EPS)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + _EPS)


def _palm_local(p):
    """Express points in the hand's own frame (..., 21, 3) -> (..., 21, 3).

    y: wrist -> middle knuckle, x: across the knuckles (index -> pinky),
    z: palm normal. The result does not change if the hand is rotated, so it
    describes handshape independent of orientation.
    """
    y = _unit(p[..., 9, :] - p[..., 0, :])
    across = p[..., 17, :] - p[..., 5, :]
    x = _unit(across - (across * y).sum(-1, keepdims=True) * y)
    z = np.cross(x, y)
    axes = np.stack([x, y, z], axis=-2)              # (..., 3, 3)
    return np.einsum("...ij,...kj->...ki", axes, p)


def points_to_features(points):
    """Canonical points (..., 21, 3) -> engineered features (..., NUM_FEATURES).

    Works on a single hand or a whole batch.
    """
    p = np.asarray(points, dtype=np.float32)
    lead = p.shape[:-2]

    raw = p.reshape(*lead, NUM_POINT_VALUES)
    tip_dists = [np.linalg.norm(p[..., i, :] - p[..., j, :], axis=-1)
                 for i, j in combinations(_TIPS, 2)]
    joint_angles = [_angle(p[..., c[k - 1], :], p[..., c[k], :], p[..., c[k + 1], :])
                    for c in _CHAINS for k in (1, 2, 3)]
    dirs = [p[..., tip, :] - p[..., base, :] for base, tip in _FINGER_DIRS]
    spread = [_angle(dirs[i], np.zeros_like(dirs[i]), dirs[i + 1]) for i in range(4)]
    thumb = [np.linalg.norm(p[..., 4, :] - p[..., r, :], axis=-1)
             for r in _THUMB_REFS + _THUMB_DIPS]

    # Palm-frame view: rotation-invariant, and it makes "where is the thumb
    # relative to the knuckles" explicit (A / S / T / N / M / E).
    local = _palm_local(p)
    width = local[..., 17, 0] - local[..., 5, 0]
    width = np.where(np.abs(width) < _EPS, _EPS, width)
    thumb_gap = [(local[..., t, 0] - local[..., 5, 0]) / width for t in (4, 3)]

    extra = np.stack(tip_dists + joint_angles + spread + thumb + thumb_gap, axis=-1)
    palm = local.reshape(*lead, NUM_POINT_VALUES)
    return np.concatenate([raw, extra, palm], axis=-1).astype(np.float32)


NUM_FEATURES = points_to_features(np.zeros((NUM_LANDMARKS, 3))).shape[-1]


def _rotation_matrices(ax, ay, az):
    n = len(ax)
    cx, sx, cy, sy, cz, sz = (np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay),
                              np.cos(az), np.sin(az))
    rx, ry, rz = (np.zeros((n, 3, 3)) for _ in range(3))
    rx[:, 0, 0] = 1; rx[:, 1, 1] = cx; rx[:, 1, 2] = -sx; rx[:, 2, 1] = sx; rx[:, 2, 2] = cx
    ry[:, 1, 1] = 1; ry[:, 0, 0] = cy; ry[:, 0, 2] = sy; ry[:, 2, 0] = -sy; ry[:, 2, 2] = cy
    rz[:, 2, 2] = 1; rz[:, 0, 0] = cz; rz[:, 0, 1] = -sz; rz[:, 1, 0] = sz; rz[:, 1, 1] = cz
    return rz @ ry @ rx


def augment_points(points, rng, copies=2, max_rot_deg=15.0, max_tilt_deg=10.0,
                   noise=0.02):
    """Randomly rotated / tilted / jittered copies of canonical points.

    Returns (augmented_points, source_index) so labels and groups can be
    repeated with y[source_index]. Mirroring is unnecessary because left
    hands are already canonicalized to right hands.
    """
    points = np.asarray(points, dtype=np.float32)
    n = len(points)
    out = []
    for _ in range(copies):
        ax = np.radians(rng.uniform(-max_tilt_deg, max_tilt_deg, n))
        ay = np.radians(rng.uniform(-max_tilt_deg, max_tilt_deg, n))
        az = np.radians(rng.uniform(-max_rot_deg, max_rot_deg, n))
        rotated = np.einsum("nij,nkj->nki", _rotation_matrices(ax, ay, az), points)
        jitter = rng.normal(0.0, noise, points.shape)
        jitter[:, 0, :] = 0.0  # wrist stays at the origin
        out.append((rotated + jitter).astype(np.float32))
    return np.concatenate(out), np.tile(np.arange(n), copies)


def read_frame(camera):
    """Read a frame, mirror it (selfie view). Returns None on failure.

    We mirror BEFORE running MediaPipe, in both collection and recognition,
    so the model always sees the same orientation (and MediaPipe's
    Left/Right labels refer to the signer's actual hands).
    """
    success, frame = camera.read()
    if not success:
        return None
    return cv2.flip(frame, 1)


def detect(hands, frame):
    """Run MediaPipe on a BGR frame. Returns a Detection, or None if no hand."""
    height, width = frame.shape[:2]
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        return None

    landmarks = results.multi_hand_landmarks[0]
    handed = results.multi_handedness[0].classification[0]

    # x/y are normalized by image width/height, so rescale to a common unit
    # first; otherwise a non-square frame distorts hand proportions.
    raw = [[l.x * width, l.y * height, l.z * width] for l in landmarks.landmark]
    points = normalize_points(raw, handed.label)
    if points is None:
        return None
    return Detection(landmarks, points, points_to_features(points),
                     handed.label, float(handed.score))
