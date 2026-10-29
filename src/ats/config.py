"""Shared constants. Every layer reads its settings from here so one change applies everywhere."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# The seven target classes, in a fixed order used for model outputs.
CLASSES = [
    "lying_down",
    "sitting",
    "standing_in_place",
    "standing_moving",
    "walking",
    "running",
    "bicycling",
]
DISPLAY = {
    "lying_down": "Lying down",
    "sitting": "Sitting",
    "standing_in_place": "Standing in place",
    "standing_moving": "Standing and moving",
    "walking": "Walking",
    "running": "Running",
    "bicycling": "Bicycling",
}
# ExtraSensory original-label columns -> our class names.
LABEL_COLUMNS = {
    "original_label:LYING_DOWN": "lying_down",
    "original_label:SITTING": "sitting",
    "original_label:STANDING_IN_PLACE": "standing_in_place",
    "original_label:STANDING_AND_MOVING": "standing_moving",
    "original_label:WALKING": "walking",
    "original_label:RUNNING": "running",
    "original_label:BICYCLING": "bicycling",
}

CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

# Preprocessing
FS = 25                 # Hz, common time base required by the brief
WIN_S = 6.0             # analysis window length (s); 150 samples, >=6 cycles at 1 Hz
HOP_S = 3.0             # window hop (s); 50% overlap, a whole number of samples
MIN_COVERAGE = 0.6      # fraction of a window that must hold real samples
MAX_INTERP_GAP_S = 0.2  # gaps longer than this stay missing instead of interpolated

# Aggregation
SMOOTHING = "hmm"       # "hmm" (Viterbi decoding) or "moving_average" (the first baseline)
SMOOTH_WINDOWS = 5      # moving-average length over window probabilities
# Sticky HMM, tuned on held-out TRAINING users by scripts/tune_smoothing.py (test users unseen).
# tau 600-4800 s forms a flat optimum (results/extrasensory/smoothing_tuning.json); 600 s is
# picked as the interior, physically plausible value.
HMM_TAU_S = 600.0       # expected bout length (s); sets the cost of switching activity
HMM_PRESMOOTH = 5       # moving average applied before decoding (1 = none)
HMM_PRIOR_WEIGHT = 0.5  # weight of the log class prior added to the emissions
CLASS_PRIOR = {         # share of labelled windows per class among the tuning script's fit users
    "lying_down": 0.3692, "sitting": 0.4300, "standing_in_place": 0.0212, "standing_moving": 0.0962,
    "walking": 0.0613, "running": 0.0037, "bicycling": 0.0184,
}
MERGE_GAP_S = 45.0      # bridge gaps up to this long between same-label windows
                        # (ExtraSensory records ~20 s per minute, so gaps of ~40 s are normal)
MIN_INTERVAL_S = 9.0    # shorter intervals are treated as flicker
MIN_BOUT_S = 10.0       # shortest bout that counts as an occurrence
PROLONGED_S = 600.0     # "prolonged" = at least 10 minutes

TIME_BASE = "seconds from start of recording"
