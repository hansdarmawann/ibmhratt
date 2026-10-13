"""Shared paths, schema, and reproducible experiment settings."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ROOT_ENV, DATA_ENV, MODELS_ENV = "ATTRITION_ROOT", "ATTRITION_DATA_PATH", "ATTRITION_MODELS_DIR"
DATA_RELATIVE = Path("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
_PACKAGE_DIR = Path(__file__).resolve().parent
# src/attrition/config.py -> checkout root. An installed wheel has no checkout and uses the working directory.
CHECKOUT_ROOT = _PACKAGE_DIR.parents[1] if _PACKAGE_DIR.parent.name == "src" else None


def normalize_path(value: str) -> Path:
    """An absolute path from operator input (environment or CLI); `~` expands, relative paths use the cwd.

    Only normalizes: nothing is read or created here. Readers check existence, and
    only training and pruning write.
    """
    value = value.strip()
    if not value or "\0" in value:
        raise ValueError("Path must be nonempty and must not contain NUL characters.")
    return Path(value).expanduser().resolve()


def env_path(name: str, environ: Mapping[str, str] = os.environ) -> Path | None:
    """A normalized path from an environment variable; unset or blank means no override."""
    value = environ.get(name, "")
    if not value.strip():
        return None
    try:
        return normalize_path(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {exc}") from exc


@dataclass(frozen=True)
class Paths:
    """Data and artifact locations, set by trusted operators only, never by API requests."""

    root: Path
    data: Path
    models: Path

    @property
    def current_run(self) -> Path:
        return self.models / "current.json"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> "Paths":
        """Checkout defaults, overridden by ATTRITION_ROOT, ATTRITION_DATA_PATH, and ATTRITION_MODELS_DIR."""
        root = env_path(ROOT_ENV, environ) or CHECKOUT_ROOT or Path.cwd().resolve()
        return cls(root, env_path(DATA_ENV, environ) or root / DATA_RELATIVE,
                   env_path(MODELS_ENV, environ) or root / "models")


PATHS = Paths.from_env()
ROOT = PATHS.root
DATA_PATH = PATHS.data
MODELS_DIR = PATHS.models
MODEL_PATH = MODELS_DIR / "attrition_pipeline.joblib"
CURRENT_RUN = PATHS.current_run
METRICS_DIR = ROOT / "reports/metrics"
FIGURES_DIR = ROOT / "reports/figures"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
TARGET = "Attrition"
TARGET_MAPPING = {"No": 0, "Yes": 1}
DEFAULT_THRESHOLD = 0.50
CONSTANT_COLUMNS = {"EmployeeCount": 1, "Over18": "Y", "StandardHours": 80}
ID_COLUMN = "EmployeeNumber"
SENSITIVE_COLUMNS = ["Gender", "Age", "MaritalStatus"]
# Used only in an explicitly reported training-CV ablation experiment.
REDUNDANT_COLUMNS = ["JobLevel", "YearsInCurrentRole", "YearsWithCurrManager"]
CATEGORIES = {
    "BusinessTravel": ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
    "Department": ["Sales", "Research & Development", "Human Resources"],
    "EducationField": ["Life Sciences", "Other", "Medical", "Marketing",
                       "Technical Degree", "Human Resources"],
    "Gender": ["Female", "Male"],
    "JobRole": ["Sales Executive", "Research Scientist", "Laboratory Technician",
                "Manufacturing Director", "Healthcare Representative", "Manager",
                "Sales Representative", "Research Director", "Human Resources"],
    "MaritalStatus": ["Single", "Married", "Divorced"],
    "OverTime": ["Yes", "No"],
}
# Broad semantic bounds, not fitted dataset minima/maxima. No clipping is applied.
NUMERIC_BOUNDS = {
    "Age": (18, 100), "DailyRate": (0, None), "DistanceFromHome": (0, None),
    "Education": (1, 5), "EnvironmentSatisfaction": (1, 4),
    "HourlyRate": (0, None), "JobInvolvement": (1, 4), "JobLevel": (1, 5),
    "JobSatisfaction": (1, 4), "MonthlyIncome": (0, None),
    "MonthlyRate": (0, None), "NumCompaniesWorked": (0, None),
    "PercentSalaryHike": (0, 100), "PerformanceRating": (1, 4),
    "RelationshipSatisfaction": (1, 4), "StockOptionLevel": (0, 3),
    "TotalWorkingYears": (0, 82), "TrainingTimesLastYear": (0, None),
    "WorkLifeBalance": (1, 4), "YearsAtCompany": (0, 82),
    "YearsInCurrentRole": (0, 82), "YearsSinceLastPromotion": (0, 82),
    "YearsWithCurrManager": (0, 82),
}
FEATURE_COLUMNS = list(NUMERIC_BOUNDS) + list(CATEGORIES)
REQUIRED_COLUMNS = FEATURE_COLUMNS + list(CONSTANT_COLUMNS) + [ID_COLUMN, TARGET]
DISCLAIMER = (
    "Educational analysis only. This project must not be used as an automated "
    "system for firing, promotion, hiring, disciplinary action, or other "
    "high-impact employment decisions. Predictions are analytical signals, "
    "not causal evidence or reliable forecasts for individuals."
)
