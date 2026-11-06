"""Train and evaluate a fingerspelling classifier.

    python train.py                       # compare models with session-grouped CV
    python train.py --holdout alice_day2  # also test on a session never used
    python train.py --model svm           # skip the comparison

Evaluation is split BY SESSION: frames from one recording are near-duplicates,
so a random split would leak and overstate accuracy. The number printed here
is how well the model does on a recording session it has never seen.

    pip install -r requirements.txt
"""
import argparse
import datetime
import sys
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
from hand_features import (FEATURE_VERSION, NUM_LANDMARKS, POINT_COLUMNS,
                           augment_points, points_to_features)

MIN_SAMPLES_WARN = 50


def candidates():
    return {
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000)),
        "svm": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
        "mlp": make_pipeline(StandardScaler(),
                             MLPClassifier((128, 64), max_iter=500,
                                           early_stopping=True, random_state=42)),
        "forest": RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                         random_state=42),
    }


def load_dataset(path):
    if not path.exists():
        sys.exit(f"{path} not found. Run collect_data.py first.")
    df = pd.read_csv(path, keep_default_na=False,
                     dtype={"label": str, "session": str})
    if "label" not in df.columns or "session" not in df.columns:
        sys.exit(f"{path} is in the old format (no header/session column). "
                 "Move it aside and re-collect with the new collect_data.py.")
    pts = df[POINT_COLUMNS].to_numpy(np.float32).reshape(-1, NUM_LANDMARKS, 3)
    return pts, df["label"].to_numpy(), df["session"].to_numpy()


def build_training_set(pts, y, groups, rng, copies):
    """Add augmented copies (training data only), then compute features."""
    if copies > 0:
        aug, src = augment_points(pts, rng, copies)
        pts = np.concatenate([pts, aug])
        y = np.concatenate([y, y[src]])
        groups = np.concatenate([groups, groups[src]])
    return points_to_features(pts), y, groups


def make_splits(y, groups, max_splits):
    """Session-grouped folds when possible, stratified random otherwise."""
    n_sessions = len(np.unique(groups))
    dummy = np.zeros(len(y))
    if n_sessions >= 2:
        n = min(max_splits, n_sessions)
        return list(GroupKFold(n_splits=n).split(dummy, y, groups)), True
    return list(StratifiedKFold(max_splits, shuffle=True,
                                random_state=42).split(dummy, y)), False


def cv_predict(model, pts, X, y, groups, splits, rng, copies):
    oof = np.empty(len(y), dtype=y.dtype)
    for train_idx, test_idx in splits:
        X_train, y_train, _ = build_training_set(
            pts[train_idx], y[train_idx], groups[train_idx], rng, copies)
        oof[test_idx] = clone(model).fit(X_train, y_train).predict(X[test_idx])
    return oof


def fit_calibrated(base, pts, y, groups, rng, copies):
    """Fit on augmented data, with probabilities calibrated on held-out folds.

    Raw model scores (especially random forests) are not real probabilities,
    so a MIN_CONF threshold means little until they are calibrated.
    """
    X, yy, gg = build_training_set(pts, y, groups, rng, copies)
    splits, _ = make_splits(yy, gg, 3)
    try:
        cal = CalibratedClassifierCV(clone(base), cv=splits, method="sigmoid",
                                     ensemble=False)
        return cal.fit(X, yy)
    except Exception as exc:  # e.g. a fold is missing a class
        print(f"Grouped calibration failed ({exc}); using plain 3-fold instead.")
        cal = CalibratedClassifierCV(clone(base), cv=3, method="sigmoid",
                                     ensemble=False)
        return cal.fit(X, yy)


def single_thread(cal):
    """Parallelism only adds overhead for one-sample predictions."""
    try:
        for cc in cal.calibrated_classifiers_:
            est = cc.estimator
            est.set_params(**{k: 1 for k in est.get_params(deep=True)
                              if k.endswith("n_jobs")})
    except AttributeError:
        pass


def most_confused(y_true, y_pred, top=8):
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pairs = [(labels[i], labels[j], int(cm[i, j]))
             for i in range(len(labels)) for j in range(len(labels))
             if i != j and cm[i, j] > 0]
    return sorted(pairs, key=lambda t: -t[2])[:top]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(candidates()),
                        help="train only this model instead of comparing all")
    parser.add_argument("--holdout", nargs="*", default=[],
                        help="session name(s) kept out of training and selection")
    parser.add_argument("--aug-copies", type=int, default=2,
                        help="augmented copies per training sample (default 2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    pts, y, groups = load_dataset(config.DATA_FILE)
    sessions = sorted(set(groups))
    counts = Counter(y)
    print(f"Loaded {len(y)} samples, {len(counts)} labels, {len(sessions)} sessions")
    sparse = sorted(l for l, c in counts.items() if c < MIN_SAMPLES_WARN)
    if sparse:
        print(f"WARNING: fewer than {MIN_SAMPLES_WARN} samples for: {', '.join(sparse)}")

    hold = np.isin(groups, args.holdout)
    if args.holdout and not hold.any():
        sys.exit(f"No samples found for holdout session(s) {args.holdout}. "
                 f"Sessions: {sessions}")
    dev = ~hold
    p_dev, y_dev, g_dev = pts[dev], y[dev], groups[dev]

    splits, grouped = make_splits(y_dev, g_dev, 4)
    if grouped:
        print(f"Cross-validating across {len(splits)} session-grouped folds.")
    else:
        print("WARNING: only one session, so using a random split. Frames from "
              "one recording are near-duplicates and this score will be "
              "optimistic. Record more sessions.")

    X_dev = points_to_features(p_dev)
    models = candidates()
    names = [args.model] if args.model else list(models)

    results = {}
    for name in names:
        oof = cv_predict(models[name], p_dev, X_dev, y_dev, g_dev, splits,
                         rng, args.aug_copies)
        acc = accuracy_score(y_dev, oof)
        f1 = f1_score(y_dev, oof, average="macro", zero_division=0)
        results[name] = (acc, f1, oof)
        print(f"  {name:8s} accuracy={acc:.3f}  macro-F1={f1:.3f}")

    best = max(results, key=lambda n: results[n][1])
    acc, f1, oof = results[best]
    print(f"\nBest by macro-F1: {best}")
    print(classification_report(y_dev, oof, zero_division=0))
    print("Most confused pairs (true, predicted, count):", most_confused(y_dev, oof))

    if hold.any():
        cal = fit_calibrated(models[best], p_dev, y_dev, g_dev, rng, args.aug_copies)
        pred = cal.predict(points_to_features(pts[hold]))
        print(f"\nHOLDOUT {args.holdout}: accuracy="
              f"{accuracy_score(y[hold], pred):.3f}  macro-F1="
              f"{f1_score(y[hold], pred, average='macro', zero_division=0):.3f}")
        print(classification_report(y[hold], pred, zero_division=0))

    # Final model: all data (including holdout), calibrated, with metadata.
    final = fit_calibrated(models[best], pts, y, groups, rng, args.aug_copies)
    single_thread(final)
    bundle = {
        "model": final,
        "classes": [str(c) for c in final.classes_],
        "feature_version": FEATURE_VERSION,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "base_model": best,
        "cv_accuracy": float(acc),
        "cv_macro_f1": float(f1),
        "n_samples": int(len(y)),
        "sessions": sessions,
    }
    joblib.dump(bundle, config.MODEL_FILE)
    print(f"\nSaved {config.MODEL_FILE} ({best}, CV accuracy {acc:.3f})")


if __name__ == "__main__":
    main()
