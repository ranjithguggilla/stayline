import pandas as pd

from stayline.config import CONSTANT_COLUMNS, ID_COLUMN
from stayline.data import feature_columns, survival_arrays, xy


def test_load_shape_and_target(df):
    assert len(df) == 1470
    assert set(df.Attrition.unique()) == {0, 1}
    assert df.Attrition.sum() == 237


def test_constants_and_id_dropped(df):
    for c in CONSTANT_COLUMNS + [ID_COLUMN]:
        assert c not in df.columns


def test_feature_types_cover_everything(df):
    num, cat = feature_columns(df)
    assert set(num) | set(cat) == set(df.columns) - {"Attrition"}
    assert "OverTime" in cat and "MonthlyIncome" in num


def test_split_is_stratified_and_disjoint(parts, df):
    train, test, _, _ = parts
    assert len(train) + len(test) == len(df)
    assert abs(train.Attrition.mean() - test.Attrition.mean()) < 0.01


def test_xy_drops_target(parts):
    train = parts[0]
    X, y = xy(train)
    assert "Attrition" not in X.columns and len(X) == len(y)


def test_survival_arrays_no_zero_time(df):
    ev, t = survival_arrays(df)
    assert t.min() >= 0.5 and ev.dtype == bool
    assert isinstance(df, pd.DataFrame)
