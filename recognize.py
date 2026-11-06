"""Live fingerspelling recognition.

    python recognize.py

Controls:
  hold a sign steady (~0.7s) to type that letter; keep holding to repeat it
  SPACE  add a space      BACKSPACE  delete last char
  c      clear text       ESC or q   quit
"""
import argparse
import time
import warnings
from collections import Counter, deque

import cv2
import joblib
import sklearn

import config
from hand_features import (FEATURE_VERSION, create_hands, detect, mp_drawing,
                           mp_hands, read_frame)

KEY_ESC = 27


def load_bundle(path):
    try:
        bundle = joblib.load(path)
    except FileNotFoundError:
        raise SystemExit(f"{path} not found. Run train.py first.")
    if not isinstance(bundle, dict) or bundle.get("feature_version") != FEATURE_VERSION:
        raise SystemExit(f"{path} was built with different features "
                         f"(need version {FEATURE_VERSION}). Re-run train.py.")
    if bundle.get("sklearn_version") != sklearn.__version__:
        warnings.warn(f"Model trained with scikit-learn {bundle.get('sklearn_version')}, "
                      f"running {sklearn.__version__}.")
    return bundle


def put(frame, text, org, scale=0.8, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    args = parser.parse_args()

    bundle = load_bundle(config.MODEL_FILE)
    model, classes = bundle["model"], bundle["classes"]
    print(f"Loaded {bundle['base_model']} (CV accuracy {bundle['cv_accuracy']:.3f}, "
          f"trained {bundle['trained_at']})")

    hands = create_hands()
    camera = cv2.VideoCapture(args.camera)

    recent = deque(maxlen=config.SMOOTH_WINDOW)
    needed_votes = config.MIN_VOTE_FRACTION * config.SMOOTH_WINDOW
    text = ""
    candidate = None
    bar_start = bar_end = 0.0

    try:
        while True:
            frame = read_frame(camera)
            if frame is None:
                print("Failed to access the camera.")
                break

            now = time.monotonic()
            det = detect(hands, frame)
            top3 = []

            if det:
                mp_drawing.draw_landmarks(frame, det.landmarks, mp_hands.HAND_CONNECTIONS)
                probs = model.predict_proba(det.features[None])[0]
                order = probs.argsort()[::-1][:3]
                top3 = [(classes[i], probs[i]) for i in order]
                best = order[0]
                recent.append(classes[best] if probs[best] >= config.MIN_CONF else None)

                label, votes = Counter(recent).most_common(1)[0]
                stable = (label if votes >= needed_votes
                          and label not in (None, config.NONE_LABEL) else None)

                if stable is None:
                    candidate = None
                elif stable != candidate:
                    candidate = stable
                    bar_start, bar_end = now, now + config.HOLD_SECONDS
                elif now >= bar_end:
                    text += candidate.upper()
                    # Holding on keeps the bar running; a second letter (LL, EE)
                    # is typed after REPEAT_SECONDS. Dropping the hand or
                    # changing the sign resets everything.
                    bar_start, bar_end = now, now + config.REPEAT_SECONDS
            else:
                recent.clear()
                candidate = None

            put(frame, f"Sign: {candidate.upper() if candidate else '-'}",
                (10, 40), 1.2, (0, 255, 0), 3)
            for i, (name, p) in enumerate(top3):
                put(frame, f"{name.upper():>4s} {p:4.0%}", (10, 80 + 25 * i),
                    0.6, (200, 200, 200), 1)
            if candidate:
                frac = min((now - bar_start) / (bar_end - bar_start), 1.0)
                cv2.rectangle(frame, (150, 20), (150 + int(200 * frac), 40),
                              (0, 255, 0), -1)
            put(frame, f"Text: {text[-40:]}", (10, frame.shape[0] - 20), 1.0)
            cv2.imshow("ASL fingerspelling", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (KEY_ESC, ord("q")):
                break
            elif key == ord(" "):
                text += " "
            elif key in (8, 127):
                text = text[:-1]
            elif key == ord("c"):
                text = ""
    finally:
        camera.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
