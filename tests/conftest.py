import numpy as np
import pandas as pd
import pytest

from stayline.data import feature_columns, load_raw, split


@pytest.fixture(scope="session")
def df():
    return load_raw()


@pytest.fixture(scope="session")
def parts(df):
    train, test = split(df)
    numeric, categorical = feature_columns(df)
    return train, test, numeric, categorical


@pytest.fixture()
def toy():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * x - 1.5)))).astype(int)
    return pd.DataFrame({"x": x, "g": rng.choice(["a", "b"], n), "Attrition": y})
