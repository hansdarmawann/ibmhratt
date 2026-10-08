"""Headless matplotlib plotting utilities used by CLI and notebooks."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})


def save_figure(fig, path: Path) -> None:
    """Save a layout-adjusted figure and release its memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_eda_plots(X: pd.DataFrame, y: pd.Series, directory: Path) -> None:
    """Plot only training data; counts and group sizes remain explicit."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = y.value_counts().reindex([0, 1])
    ax.bar(["No / retention", "Yes / attrition"], counts, color=["#357b89", "#d27a43"])
    ax.set(ylabel="Training observations", title="Training target distribution")
    for i, value in enumerate(counts):
        ax.text(i, value, f"{value} ({value / len(y):.1%})", ha="center", va="bottom")
    ax.set_ylim(0, max(counts) * 1.15)
    save_figure(fig, directory / "attrition_distribution.png")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for col, ax in zip(["Age", "MonthlyIncome", "DistanceFromHome", "TotalWorkingYears",
                        "YearsAtCompany", "DailyRate"], axes.flat, strict=True):
        for label, color in [(0, "#357b89"), (1, "#d27a43")]:
            ax.hist(X.loc[y == label, col], bins=16, density=True, alpha=0.55,
                    label="Yes" if label else "No", color=color)
        ax.set(title=col, ylabel="Density")
    axes.flat[0].legend(title="Attrition")
    fig.suptitle("Training numeric distributions by observed attrition", y=1.02)
    save_figure(fig, directory / "numeric_distributions.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for col, ax in zip(["OverTime", "BusinessTravel", "JobRole"], axes, strict=True):
        table = pd.DataFrame({col: X[col], "target": y}).groupby(col)["target"].agg(["mean", "size"])
        ax.barh([f"{v} (n={n})" for v, n in zip(table.index, table["size"], strict=True)], table["mean"], color="#357b89")
        ax.set(title=col, xlabel="Observed attrition rate", xlim=(0, max(0.5, table["mean"].max() + 0.05)))
    save_figure(fig, directory / "categorical_attrition.png")

    columns = ["Age", "JobLevel", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany",
               "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"]
    corr = X[columns].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(columns)), columns, rotation=55, ha="right")
    ax.set_yticks(range(len(columns)), columns)
    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.iloc[i, j]) > 0.65 else "black")
    ax.set_title("Training correlations: related career and compensation measures")
    fig.colorbar(im, ax=ax, shrink=0.75)
    save_figure(fig, directory / "numeric_correlations.png")


def save_evaluation_plots(y, probabilities: dict, selected: str, metrics: dict,
                          thresholds: pd.DataFrame, directory: Path) -> None:
    """Persist holdout curves, final confusion matrix, and OOF threshold curve."""
    for kind, display in [("roc_curve", RocCurveDisplay), ("precision_recall_curve", PrecisionRecallDisplay)]:
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, proba in probabilities.items():
            display.from_predictions(y, proba, name=name.replace("_", " "), ax=ax)
        if kind == "precision_recall_curve":
            ax.axhline(np.mean(y), color="grey", linestyle="--", label="Holdout prevalence")
        else:
            ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set_title("Final holdout model comparison")
        ax.legend(fontsize=8)
        save_figure(fig, directory / f"{kind}.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(np.asarray(metrics["confusion_matrix"]),
                           display_labels=["No", "Yes"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{selected.replace('_', ' ')} | threshold {metrics['threshold']:.2f}")
    save_figure(fig, directory / "confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in ["precision", "recall", "f1", "f2"]:
        ax.plot(thresholds["threshold"], thresholds[metric], marker=".", label=metric.upper())
    ax.axvline(metrics["threshold"], color="grey", linestyle="--", label="Selected on OOF F2")
    ax.set(xlabel="Decision threshold", ylabel="Score", ylim=(0, 1.03),
           title="Training out-of-fold threshold analysis")
    ax.legend()
    save_figure(fig, directory / "threshold_analysis.png")


def importance_plot(table: pd.DataFrame, value: str, path: Path, title: str) -> None:
    """Save top readable feature names; signed values retain their direction."""
    selected = table.reindex(table[value].abs().sort_values(ascending=False).index).head(15)
    selected = selected.sort_values(value)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(selected["feature"], selected[value], color=np.where(selected[value] >= 0, "#357b89", "#d27a43"))
    ax.set(title=title, xlabel=value.replace("_", " "))
    save_figure(fig, path)


def save_calibration_plot(bins: pd.DataFrame, directory: Path) -> None:
    """Reliability on outer-OOF training predictions; empty bins stay in the CSV."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for method, table in bins.groupby("method", sort=False):
        observed = table.loc[table.n > 0]
        ax.plot(observed.mean_probability, observed.observed_rate, marker="o", label=method)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed attrition rate",
           xlim=(0, 1), ylim=(0, 1), title="Training outer-OOF reliability (10 equal-width bins)")
    ax.legend()
    save_figure(fig, directory / "calibration_reliability.png")
