"""Model builders. Every transform lives inside the pipeline, so cross-validation never sees test folds."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import SEED

# lower rank = simpler, used by the one-standard-error rule
COMPLEXITY = {"prior": 0, "logistic": 1, "logistic_balanced": 1, "forest": 2, "boosting": 3}


def preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        verbose_feature_names_out=False,
    )


def build(name: str, numeric: list[str], categorical: list[str], seed: int = SEED) -> Pipeline:
    pre = preprocessor(numeric, categorical)
    if name == "prior":
        est = DummyClassifier(strategy="prior")
    elif name == "logistic":
        est = LogisticRegressionCV(Cs=12, cv=5, scoring="neg_log_loss", max_iter=2000, random_state=seed)
    elif name == "logistic_balanced":
        est = LogisticRegressionCV(
            Cs=12, cv=5, scoring="neg_log_loss", max_iter=2000, class_weight="balanced", random_state=seed
        )
    elif name == "forest":
        est = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    elif name == "boosting":
        est = HistGradientBoostingClassifier(
            learning_rate=0.04, max_iter=160, max_depth=3, min_samples_leaf=20, l2_regularization=1.0, random_state=seed
        )
    else:
        raise ValueError(f"unknown model {name!r}")
    return Pipeline([("pre", pre), ("model", est)])
