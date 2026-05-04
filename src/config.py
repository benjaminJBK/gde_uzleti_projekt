from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "train.csv"

RANDOM_STATE = 42