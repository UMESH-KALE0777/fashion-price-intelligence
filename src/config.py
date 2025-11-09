from pathlib import Path

# Project root = folder that contains main.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_DATA_PATH = DATA_DIR / "raw_data.csv"
CLEAN_DATA_PATH = DATA_DIR / "clean_data.csv"
PREDICTIONS_PATH = DATA_DIR / "predictions.csv"

MODEL_PATH = MODELS_DIR / "trained_model.cbm"
METADATA_PATH = MODELS_DIR / "metadata.pkl"

TARGET_COLUMN = "price"
TEST_SIZE = 0.2
RANDOM_STATE = 42
