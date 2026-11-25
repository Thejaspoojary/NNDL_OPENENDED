"""Central configuration for the Pneumonia Detection project."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Base directories
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATA_DIR / "train"
VALIDATION_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
TEST_DATA_DIR = DATA_DIR / "test_data"

MODEL_DIR = BASE_DIR / "models"
CNN_MODEL_PATH = MODEL_DIR / "cnn_pneumonia.keras"
MOBILENETV2_MODEL_PATH = MODEL_DIR / "mobilenetv2_pneumonia.keras"
BEST_MODEL_PATH = MODEL_DIR / "best_model.keras"

RESULTS_DIR = BASE_DIR / "results"
EVALUATION_METRICS_PATH = RESULTS_DIR / "evaluation_metrics.json"
BEST_MODEL_INFO_PATH = RESULTS_DIR / "best_model.json"
HISTORY_DIR = RESULTS_DIR / "history"
PLOTS_DIR = RESULTS_DIR / "plots"

LOG_DIR = BASE_DIR / "logs"

# ---------------------------------------------------------------------------
# Image / model settings
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
NORMAL_CLASS = CLASS_NAMES[0]
PNEUMONIA_CLASS = CLASS_NAMES[1]
NUM_CLASSES = len(CLASS_NAMES)
POSITIVE_CLASS_INDEX = 1  # PNEUMONIA

MODEL_CHOICES = ["cnn", "mobilenetv2"]
DEFAULT_MODEL = "cnn"

RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.3
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_TOP_LAYERS = 20
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3

# ---------------------------------------------------------------------------
# Upload / safety settings
# ---------------------------------------------------------------------------
MAX_UPLOAD_SIZE_MB = 5
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
PLOTS_ENABLED = True
