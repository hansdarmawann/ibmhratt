"""Prespecified domain features for a training-CV ablation; the served model does not use them.

Each feature is a fixed formula of existing columns, so nothing is learned from
data here; imputation and scaling still happen inside the fold-fitted pipeline.
Denominators add one year or one company because zeros occur in the dataset.
"""

import pandas as pd

ENGINEERED_FEATURES = {
    "IncomePerJobLevel": "MonthlyIncome / JobLevel",
    "CompanyShareOfCareer": "YearsAtCompany / (TotalWorkingYears + 1)",
    "YearsPerPriorEmployer": "TotalWorkingYears / (NumCompaniesWorked + 1)",
    "PromotionWaitShare": "YearsSinceLastPromotion / (YearsAtCompany + 1)",
    "MeanSatisfaction": "mean of Environment, Job, Relationship satisfaction and WorkLifeBalance",
}
SATISFACTION_COLUMNS = ["EnvironmentSatisfaction", "JobSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance"]


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Append ENGINEERED_FEATURES; missing inputs give missing outputs for the pipeline imputer."""
    def number(column: str) -> pd.Series:
        return pd.to_numeric(frame[column], errors="coerce")

    result = frame.copy()
    result["IncomePerJobLevel"] = number("MonthlyIncome") / number("JobLevel")
    result["CompanyShareOfCareer"] = number("YearsAtCompany") / (number("TotalWorkingYears") + 1)
    result["YearsPerPriorEmployer"] = number("TotalWorkingYears") / (number("NumCompaniesWorked") + 1)
    result["PromotionWaitShare"] = number("YearsSinceLastPromotion") / (number("YearsAtCompany") + 1)
    result["MeanSatisfaction"] = pd.concat([number(c) for c in SATISFACTION_COLUMNS], axis=1).mean(axis=1)
    return result
