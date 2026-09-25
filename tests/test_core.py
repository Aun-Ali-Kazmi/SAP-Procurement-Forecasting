"""Fast unit tests for the parts that do not need the deep-learning stack.

    python -m pytest -q
"""

import numpy as np
import pandas as pd

from forecasting.data import build_series, data_end_month
from forecasting.evaluation import fold_origins
from forecasting.metrics import mape, mase, smape
from forecasting.models import croston_sba, naive


def _raw():
    return pd.DataFrame({
        "Plant": ["A", "A", "B", "A"],
        "Material": ["1", "1", "1", "2"],
        "Document Date": pd.to_datetime(["2024-01-10", "2024-01-20", "2024-03-28", "2024-02-01"]),
        "Net Order Value": [1_000_000.0, 500_000.0, 2_000_000.0, 7.0],
    })


def test_build_series_fills_zero_months_and_scales():
    s = build_series(_raw(), "1", 1_000_000)
    assert s["ds"].dt.strftime("%Y-%m").tolist() == ["2024-01", "2024-02", "2024-03"]
    assert s["y"].tolist() == [1.5, 0.0, 2.0]


def test_build_series_plant_filter():
    s = build_series(_raw(), "1", 1_000_000, plant="B")
    assert s["y"].tolist() == [2.0]


def test_no_synthetic_history():
    s = build_series(_raw(), "1", 1_000_000)
    assert s["ds"].min() == pd.Timestamp("2024-01-01")  # starts at the first real purchase


def test_partial_last_month_excluded():
    raw = _raw()
    assert data_end_month(raw) == pd.Timestamp("2024-03-01")
    raw.loc[2, "Document Date"] = pd.Timestamp("2024-03-10")  # export stops early in March
    assert data_end_month(raw) == pd.Timestamp("2024-02-01")


def test_fold_origins_last_fold_is_holdout():
    o = fold_origins(n=20, h=3, max_origins=6, min_train=8)
    assert o[-1] == 17 and len(o) == 6
    assert fold_origins(n=14, h=3, max_origins=6, min_train=8) == [8, 9, 10, 11]


def test_metrics_handle_zero_actuals():
    assert np.isnan(mape([0, 0], [1, 2]))
    assert smape([0, 0], [0, 0]) == 0.0
    assert mase([2, 2], [1, 1], [0, 2, 0, 2]) == 0.5


def test_benchmarks():
    train = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=6, freq="MS"),
                          "y": [0, 3, 0, 0, 3, 0.0]})
    assert naive(train, 2, 80, 1, {}).yhat.tolist() == [0.0, 0.0]
    fc = croston_sba(train, 3, 80, 1, {})
    assert len(fc.yhat) == 3 and fc.yhat[0] > 0 and fc.lo is None
