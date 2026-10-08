"""Build fold-fitted numeric and categorical preprocessing pipelines."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from attrition.config import CONSTANT_COLUMNS, FEATURE_COLUMNS, ID_COLUMN
from attrition.features.engineering import ENGINEERED_FEATURES, add_engineered_features


def normalize_missing(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize pandas/JSON missing sentinels without learning or mutation."""
    result = frame.copy().astype(object)
    return result.where(pd.notna(result), np.nan)


def feature_exclusions(X_train: pd.DataFrame) -> dict[str, str]:
    """Confirm dataset-specific constants on training rows before excluding."""
    reasons = {ID_COLUMN: "Unique employee identifier; no portable predictive meaning."}
    for col, expected in CONSTANT_COLUMNS.items():
        if col in X_train and (X_train[col].nunique(dropna=False) != 1 or not X_train[col].eq(expected).all()):
            raise ValueError(f"Expected {col} to be constant {expected!r} in training data.")
        reasons[col] = f"Confirmed constant in training data ({expected!r}); no variation."
    return reasons


def build_preprocessor(X_train: pd.DataFrame, *, scale: bool = True, exclude: tuple | list = (),
                       extra_numeric: tuple | list = ()) -> ColumnTransformer:
    """Define transformations; all learned statistics are fitted within Pipeline."""
    feature_exclusions(X_train)
    columns = [c for c in FEATURE_COLUMNS if c not in exclude]
    numeric = X_train[columns].select_dtypes(include="number").columns.tolist() + list(extra_numeric)
    categorical = [c for c in columns if c not in numeric]
    numeric_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ], remainder="drop", verbose_feature_names_out=False)


def build_pipeline(X_train: pd.DataFrame, estimator, *, scale: bool = True,
                   exclude: tuple | list = (), engineered: bool = False) -> Pipeline:
    """Keep missing normalization, fitted transforms, and estimator together.

    engineered=True appends the prespecified ablation features; the served model never uses it.
    """
    steps: list[tuple[str, object]] = [("normalize", FunctionTransformer(normalize_missing))]
    extra: list[str] = []
    if engineered:
        steps.append(("engineer", FunctionTransformer(add_engineered_features)))
        extra = list(ENGINEERED_FEATURES)
    steps += [("preprocess", build_preprocessor(X_train, scale=scale, exclude=exclude, extra_numeric=extra)),
              ("model", estimator)]
    return Pipeline(steps)
