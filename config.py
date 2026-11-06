"""Shared settings for collection, training and recognition."""
from pathlib import Path

# --- Paths ---------------------------------------------------------------
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "landmarks.csv"
MODEL_FILE = Path("model.pkl")
CAMERA_INDEX = 0

# --- Labels --------------------------------------------------------------
NONE_LABEL = "none"            # relaxed hand / transition frames
SKIP_LETTERS = {"j", "z"}      # motion letters, handled later

# --- Collection ----------------------------------------------------------
SAMPLE_INTERVAL = 0.1          # seconds between auto-recorded samples
COUNTDOWN_SECONDS = 3          # time to get your hand into position
TARGET_PER_LABEL = 300         # samples per label to aim for
MIN_DETECT_SCORE = 0.8         # skip frames where MediaPipe is unsure

# --- Recognition ---------------------------------------------------------
SMOOTH_WINDOW = 10             # frames used for majority-vote smoothing
MIN_VOTE_FRACTION = 0.8        # share of the window that must agree
MIN_CONF = 0.6                 # minimum (calibrated) confidence per frame
HOLD_SECONDS = 0.7             # steady time before a letter is typed
REPEAT_SECONDS = 1.2           # keep holding this long to type it again (LL, EE)
