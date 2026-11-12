"""Gradio front end. The scoring logic lives in score.py so it can be tested without a browser."""

from __future__ import annotations

import sys

import joblib

from .config import OUT_DIR
from .score import Bundle, score_employee

FIELDS = [
    ("OverTime", ["No", "Yes"]),
    ("MonthlyIncome", None),
    ("Age", None),
    ("JobRole", None),
    ("MaritalStatus", ["Single", "Married", "Divorced"]),
    ("YearsAtCompany", None),
    ("YearsSinceLastPromotion", None),
    ("JobSatisfaction", [1, 2, 3, 4]),
    ("WorkLifeBalance", [1, 2, 3, 4]),
    ("BusinessTravel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]),
    ("StockOptionLevel", [0, 1, 2, 3]),
    ("DistanceFromHome", None),
]


def load_bundle(path=OUT_DIR / "bundle.joblib") -> Bundle:
    return joblib.load(path)


def build_app(bundle: Bundle):
    import gradio as gr

    roles = bundle.meta["job_roles"]
    inputs = []
    for name, choices in FIELDS:
        default = bundle.defaults[name]
        if name == "JobRole":
            inputs.append(gr.Dropdown(roles, value=default, label=name))
        elif choices:
            inputs.append(
                gr.Dropdown(
                    choices, value=type(choices[0])(default) if not isinstance(default, str) else default, label=name
                )
            )
        else:
            inputs.append(gr.Number(value=default, label=name))

    def run(*values):
        rec = dict(zip([f for f, _ in FIELDS], values, strict=True))
        out = score_employee(bundle, rec)
        reasons = out.pop("reasons")
        text = "\n".join(
            ["Raises risk:"]
            + [f"- {r}" for r in reasons["raises_risk"]]
            + ["Lowers risk:"]
            + [f"- {r}" for r in reasons["lowers_risk"]]
        )
        return out, text

    return gr.Interface(
        run,
        inputs,
        [gr.JSON(label="score"), gr.Textbox(label="why", lines=8)],
        title="stayline",
        description="Attrition risk and the value of a retention call. Unfilled fields use the typical employee.",
    )


def main(argv=None) -> int:
    path = OUT_DIR / "bundle.joblib"
    if not path.exists():
        print("no bundle found; run `stayline` first", file=sys.stderr)
        return 1
    build_app(load_bundle(path)).launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
