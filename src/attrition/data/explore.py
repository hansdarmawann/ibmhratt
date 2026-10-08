"""Training-partition EDA shared by notebooks and the training command."""

from pathlib import Path

import pandas as pd

from attrition.config import CATEGORIES, FEATURE_COLUMNS, TARGET
from attrition.visualization.plots import save_eda_plots


def exploration_tables(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Inspect every predictor, outliers, and correlations using training rows."""
    frame = X_train[FEATURE_COLUMNS].copy()
    frame[TARGET] = y_train
    numeric = frame[FEATURE_COLUMNS].select_dtypes(include="number")
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    iqr = q3 - q1
    outliers = ((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum()
    groups = []
    for col in FEATURE_COLUMNS:
        if col in CATEGORIES or frame[col].nunique() <= 10:
            rates = frame.groupby(col, observed=True, dropna=False)[TARGET].agg(["count", "mean"])
            rates = rates.rename(columns={"mean": "attrition_rate"}).reset_index()
            rates.columns = ["value", "count", "attrition_rate"]
            rates.insert(0, "feature", col)
            groups.append(rates)
    return {
        "summary_statistics": numeric.describe().T,
        "target_distribution": y_train.value_counts().rename_axis("attrition").to_frame("count"),
        "category_target_rates": pd.concat(groups, ignore_index=True),
        "numeric_by_target": frame.groupby(TARGET)[numeric.columns].agg(["mean", "median"]),
        "iqr_outlier_counts": outliers.rename("outlier_count").to_frame(),
        "numeric_correlations": numeric.corr(),
    }


def save_exploration(X_train: pd.DataFrame, y_train: pd.Series,
                     metrics_dir: Path, figures_dir: Path) -> dict:
    """Persist training-only tables and a compact set of informative plots."""
    tables = exploration_tables(X_train, y_train)
    for name, table in tables.items():
        table.to_csv(metrics_dir / f"eda_{name}.csv")
    save_eda_plots(X_train, y_train, figures_dir)
    return tables
