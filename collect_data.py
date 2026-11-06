"""Collect labeled hand-landmark samples.

    python collect_data.py --session alice_lamp_day1

Controls:
  a-z      select the letter to record (J and Z need motion, skipped for now)
  0        select the "none" class (relaxed hand, in-between poses)
  SPACE    start (after a countdown) / stop-and-save a take
  TAB      discard the current take, or undo the last saved take
  ESC      quit (any in-progress take is saved)

Session names are used to split train/test honestly, so make them meaningful:
one session = one person + one setting + one sitting. Collect several.
Vary hand angle, distance and position slightly while recording.
"""
import argparse
import csv
import string
import time
from collections import Counter

import cv2

import config
from hand_features import (POINT_COLUMNS, create_hands, detect, mp_drawing,
                           mp_hands, read_frame)

HEADER = ["label", "session", "hand", "score"] + POINT_COLUMNS
LABELS = ([c for c in string.ascii_lowercase if c not in config.SKIP_LETTERS]
          + [config.NONE_LABEL])
KEY_ESC, KEY_TAB = 27, 9


def load_counts():
    counts = Counter()
    if config.DATA_FILE.exists():
        with open(config.DATA_FILE, newline="") as f:
            for row in csv.reader(f):
                if row and row[0] != "label":
                    counts[row[0]] += 1
    return counts


def append_rows(rows):
    config.DATA_DIR.mkdir(exist_ok=True)
    new_file = not config.DATA_FILE.exists() or config.DATA_FILE.stat().st_size == 0
    with open(config.DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(HEADER)
        writer.writerows(rows)


def remove_last_rows(n):
    lines = config.DATA_FILE.read_text().splitlines(keepends=True)
    keep = max(1, len(lines) - n)  # never delete the header
    config.DATA_FILE.write_text("".join(lines[:keep]))


class Recorder:
    """Holds recording state; a 'take' is one SPACE-to-SPACE recording."""

    def __init__(self, session):
        self.session = session
        self.counts = load_counts()
        self.session_counts = Counter()
        self.label = None
        self.state = "idle"            # idle | countdown | recording
        self.countdown_end = 0.0
        self.take = []
        self.history = []              # (label, n_rows) of saved takes
        self.last_saved = 0.0
        self.message = ""

    def select(self, label):
        self.finish_take()
        self.label = label

    def toggle(self, now):
        if self.state == "idle":
            if self.label:
                self.state = "countdown"
                self.countdown_end = now + config.COUNTDOWN_SECONDS
        else:
            self.finish_take()

    def update(self, now, det):
        if self.state == "countdown" and now >= self.countdown_end:
            self.state = "recording"
        if (self.state == "recording" and det is not None
                and det.score >= config.MIN_DETECT_SCORE
                and now - self.last_saved >= config.SAMPLE_INTERVAL):
            self.take.append([self.label, self.session, det.hand,
                              round(det.score, 3)] + det.points.flatten().tolist())
            self.last_saved = now

    def finish_take(self):
        self.state = "idle"
        if self.take:
            append_rows(self.take)
            n = len(self.take)
            self.history.append((self.label, n))
            self.counts[self.label] += n
            self.session_counts[self.label] += n
            self.message = f"saved {n} samples of '{self.label}'"
            self.take = []

    def undo(self):
        if self.state != "idle" or self.take:
            self.take, self.state = [], "idle"
            self.message = "discarded current take"
        elif self.history:
            label, n = self.history.pop()
            remove_last_rows(n)
            self.counts[label] -= n
            self.session_counts[label] -= n
            self.message = f"undid last take ({n} samples of '{label}')"

    def current_count(self):
        return (self.counts[self.label] + len(self.take)) if self.label else 0

    def neediest(self, k=6):
        short = sorted((l for l in LABELS if self.counts[l] < config.TARGET_PER_LABEL),
                       key=lambda l: self.counts[l])
        return short[:k]


def put(frame, text, y, color=(255, 255, 255), scale=0.6, thickness=2):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", default=time.strftime("%Y%m%d_%H%M%S"),
                        help="session name (person_setting_day works well)")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    args = parser.parse_args()

    rec = Recorder(args.session)
    hands = create_hands()
    camera = cv2.VideoCapture(args.camera)
    print(f"Session: {args.session}")

    try:
        while True:
            frame = read_frame(camera)
            if frame is None:
                print("Failed to access the camera.")
                break

            det = detect(hands, frame)
            now = time.time()
            rec.update(now, det)
            if det:
                mp_drawing.draw_landmarks(frame, det.landmarks, mp_hands.HAND_CONNECTIONS)

            recording = rec.state == "recording"
            color = (0, 0, 255) if recording else (0, 255, 0)
            put(frame, f"Label: {rec.label or '-'}  [{rec.state}]  "
                       f"{rec.current_count()}/{config.TARGET_PER_LABEL}",
                30, color, 0.8)
            if rec.state == "countdown":
                put(frame, str(int(rec.countdown_end - now) + 1), 80, (0, 255, 255), 2.0, 4)
            elif recording and det is None:
                put(frame, "no hand detected", 60, (0, 165, 255))
            put(frame, f"session: {rec.session}   {rec.message}", 55 if det else 90,
                (200, 200, 200), 0.5, 1)
            put(frame, "needs: " + " ".join(rec.neediest()),
                frame.shape[0] - 40, (0, 255, 255), 0.5, 1)
            put(frame, "a-z/0: label | SPACE: record | TAB: undo | ESC: quit",
                frame.shape[0] - 15, (255, 255, 255), 0.5, 1)
            cv2.imshow("Collect ASL data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == KEY_ESC:
                break
            elif key == ord(" "):
                rec.toggle(now)
            elif key == KEY_TAB:
                rec.undo()
            elif key == ord("0"):
                rec.select(config.NONE_LABEL)
            elif 97 <= key <= 122:
                letter = chr(key)
                if letter in config.SKIP_LETTERS:
                    rec.message = f"'{letter}' needs motion; skipped for now"
                else:
                    rec.select(letter)
    finally:
        rec.finish_take()
        camera.release()
        hands.close()
        cv2.destroyAllWindows()

    print("This session:", dict(sorted(rec.session_counts.items())))
    print("All data:    ", dict(sorted(rec.counts.items())))


if __name__ == "__main__":
    main()
