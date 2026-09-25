"""Forecast accuracy metrics.

MAE is the primary per-material metric (as in the thesis). Because the three
materials are on very different spend scales, MASE (MAE divided by the
in-sample MAE of a one-step naive forecast) is used to compare and average
models *across* materials. MAPE is undefined when actual spend is zero, so it
is only computed over non-zero months and reported as missing otherwise.
"""

import numpy as np


def mae(y, f):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(f))))


def rmse(y, f):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(f)) ** 2)))


def mape(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    mask = y != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y[mask] - f[mask]) / y[mask])) * 100)


def smape(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    denom = np.abs(y) + np.abs(f)
    terms = np.where(denom == 0, 0.0, 2 * np.abs(y - f) / np.where(denom == 0, 1, denom))
    return float(np.mean(terms) * 100)


def naive_scale(y_train):
    """In-sample MAE of the one-step naive forecast (MASE denominator)."""
    y_train = np.asarray(y_train, float)
    if len(y_train) < 2:
        return np.nan
    scale = np.mean(np.abs(np.diff(y_train)))
    return float(scale) if scale > 0 else np.nan


def mase(y, f, y_train):
    scale = naive_scale(y_train)
    return mae(y, f) / scale if scale == scale else np.nan


def coverage(y, lo, hi):
    """Share of actuals inside the prediction interval (PICP)."""
    if lo is None or hi is None:
        return np.nan
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    if np.isnan(lo).all():
        return np.nan
    return float(np.mean((y >= lo) & (y <= hi)))


def all_metrics(y, f, y_train, lo=None, hi=None) -> dict:
    return {
        "MAE": mae(y, f),
        "RMSE": rmse(y, f),
        "MAPE": mape(y, f),
        "sMAPE": smape(y, f),
        "MASE": mase(y, f, y_train),
        "Bias": float(np.mean(np.asarray(f) - np.asarray(y))),
        "PICP": coverage(y, lo, hi),
    }
