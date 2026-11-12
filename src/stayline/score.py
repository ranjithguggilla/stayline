"""Scoring one employee and explaining the number in plain words."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import COSTS, Costs
from .economics import replacement_cost

PHRASES = {
    "OverTime": "works overtime",
    "MonthlyIncome": "monthly pay",
    "StockOptionLevel": "stock option level",
    "MaritalStatus": "marital status",
    "JobRole": "job role",
    "JobLevel": "job level",
    "YearsAtCompany": "years at the company",
    "YearsInCurrentRole": "years in the current role",
    "YearsSinceLastPromotion": "years since the last promotion",
    "YearsWithCurrManager": "years with the current manager",
    "TotalWorkingYears": "total working years",
    "BusinessTravel": "business travel",
    "DistanceFromHome": "commute distance",
    "JobSatisfaction": "job satisfaction score",
    "EnvironmentSatisfaction": "environment satisfaction score",
    "WorkLifeBalance": "work-life balance score",
    "JobInvolvement": "job involvement score",
    "RelationshipSatisfaction": "relationship satisfaction score",
    "NumCompaniesWorked": "number of past employers",
    "TrainingTimesLastYear": "training sessions last year",
    "Age": "age",
    "Department": "department",
    "EducationField": "education field",
    "Education": "education level",
    "Gender": "gender",
    "PercentSalaryHike": "last salary rise (%)",
    "PerformanceRating": "performance rating",
    "DailyRate": "daily rate",
    "HourlyRate": "hourly rate",
    "MonthlyRate": "monthly rate",
}


@dataclass
class Bundle:
    pipeline: object
    columns: list[str]
    defaults: dict
    income_cut: float
    meta: dict = field(default_factory=dict)


def baseline_defaults(train: pd.DataFrame, columns: list[str]) -> dict:
    """Median for numbers, mode for text: the 'typical employee' used when a field is left blank."""
    out = {}
    for c in columns:
        s = train[c]
        out[c] = s.mode().iloc[0] if not pd.api.types.is_numeric_dtype(s) else float(s.median())
    return out


def complete_record(bundle: Bundle, record: dict) -> pd.DataFrame:
    row = {c: record.get(c, bundle.defaults[c]) for c in bundle.columns}
    return pd.DataFrame([row])[bundle.columns]


def explain(bundle: Bundle, row: pd.DataFrame, k: int = 3) -> dict[str, list[str]]:
    """Swap each field for the typical value and see how far the risk moves. Model-agnostic and cheap."""
    base = float(bundle.pipeline.predict_proba(row)[0, 1])
    deltas = {}
    for c in bundle.columns:
        alt = row.copy()
        alt[c] = bundle.defaults[c]
        deltas[c] = base - float(bundle.pipeline.predict_proba(alt)[0, 1])
    ranked = sorted(deltas.items(), key=lambda kv: kv[1], reverse=True)

    def fmt(col, d, word):
        val = row[col].iloc[0]
        val = f"{val:g}" if isinstance(val, (int, float, np.floating, np.integer)) else str(val)
        return f"{PHRASES.get(col, col)} = {val} ({word} risk by {abs(d) * 100:.1f} points vs a typical employee)"

    up = [fmt(c, d, "raises") for c, d in ranked if d > 0.002][:k]
    down = [fmt(c, d, "lowers") for c, d in ranked[::-1] if d < -0.002][:k]
    return {"raises_risk": up, "lowers_risk": down}


def band(p: float, thresholds: tuple[float, float]) -> str:
    lo, hi = thresholds
    return "high" if p >= hi else "medium" if p >= lo else "low"


def score_employee(bundle: Bundle, record: dict, costs: Costs = COSTS) -> dict:
    row = complete_record(bundle, record)
    p = float(bundle.pipeline.predict_proba(row)[0, 1])
    repl = float(replacement_cost(row["MonthlyIncome"].to_numpy(), costs)[0])
    gain = p * repl * costs.save_rate - costs.cost_per_conversation
    return {
        "attrition_probability": round(p, 4),
        "risk_band": band(p, tuple(bundle.meta["band_thresholds"])),
        "replacement_cost_usd": round(repl, 0),
        "expected_net_value_of_a_call_usd": round(gain, 0),
        "worth_a_call": bool(gain > 0),
        "reasons": explain(bundle, row),
    }
