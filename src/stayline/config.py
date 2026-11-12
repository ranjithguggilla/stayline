"""Constants shared by the pipeline: paths, seeds and the cost assumptions behind the dollar figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "hr_attrition.csv"
OUT_DIR = ROOT / "outputs"
TABLE_DIR = OUT_DIR / "tables"
ASSET_DIR = ROOT / "assets"

SEED = 7
TEST_SIZE = 0.30
CV_FOLDS = 5
CV_REPEATS = 5
BOOTSTRAPS = 2000

TARGET = "Attrition"
ID_COLUMN = "EmployeeNumber"
CONSTANT_COLUMNS = ["EmployeeCount", "Over18", "StandardHours"]

# counters that stop ticking when someone leaves, so they carry part of the answer
TENURE_COLUMNS = [
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "TotalWorkingYears",
]
# pay-rate columns that look unrelated to everything else in the file
RATE_COLUMNS = ["DailyRate", "HourlyRate", "MonthlyRate"]
DEMOGRAPHIC_COLUMNS = ["Age", "Gender", "MaritalStatus"]


@dataclass(frozen=True)
class Costs:
    """Stated assumptions for the retention-call model. None of these come from the dataset."""

    replacement_share_of_annual_pay: float = 0.50
    cost_per_conversation: float = 1500.0
    save_rate: float = 0.25
    budget_fraction: float = 0.20


COSTS = Costs()
