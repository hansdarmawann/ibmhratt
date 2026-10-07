"""Shared paths, schema, and reproducible experiment settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_PATH = ROOT / "models/attrition_pipeline.joblib"
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
