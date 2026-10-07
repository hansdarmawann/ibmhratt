"""Safe CSV loading and the single reproducible holdout split."""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, RANDOM_STATE, TARGET, TARGET_MAPPING, TEST_SIZE
from src.data.validate_data import validate_data

LOGGER = logging.getLogger(__name__)


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate a CSV without modifying the source file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    try:
        frame = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        raise ValueError(f"Could not parse dataset CSV: {exc}") from exc
    audit = validate_data(frame)
    LOGGER.info("Data loaded: %s rows, %s columns; %s missing predictors",
                *frame.shape, sum(audit["missing_values"].values()))
    return frame


def split_data(frame: pd.DataFrame) -> tuple:
    """Reserve a stratified holdout before EDA or any learned transforms."""
    counts = frame[TARGET].value_counts()
    if counts.min() < 10:
        raise ValueError("Each target class needs at least 10 rows for holdout and five-fold CV.")
    X = frame.drop(columns=TARGET)
    y = frame[TARGET].map(TARGET_MAPPING).astype(int)
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                            random_state=RANDOM_STATE)
