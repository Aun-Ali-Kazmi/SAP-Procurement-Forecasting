"""Rolling-origin (expanding-window) evaluation.

For each material the forecast origin is moved forward one month at a time.
At every origin each model is trained on all months up to the origin and
forecasts the next `horizon` months, which are compared with actual spend.
The last origin is the classic hold-out split (train on everything except
the final `horizon` months). All models see exactly the same folds.

Predictions are check-pointed to CSV after every (material, model) so an
interrupted run resumes where it stopped.
"""

import logging
import time

import numpy as np
import pandas as pd

from .metrics import all_metrics
from .models import run_model

LOG = logging.getLogger("forecasting")


def fold_origins(n: int, h: int, max_origins: int, min_train: int) -> list[int]:
    """Training-set lengths for each fold; the last one is the hold-out fold."""
    last = n - h
    first = max(min_train, last - max_origins + 1)
    return list(range(first, last + 1))


def backtest(series: dict, model_names: list, cfg: dict, pred_path) -> pd.DataFrame:
    ev = cfg["evaluation"]
    h, level, seeds = ev["horizon"], ev["interval_level"], ev["seeds"]

    done = pd.read_csv(pred_path, parse_dates=["ds", "origin"]) if pred_path.exists() else pd.DataFrame()
    done_keys = set(zip(done["material"], done["model"])) if not done.empty else set()

    for material, s in series.items():
        origins = fold_origins(len(s), h, ev["max_origins"], ev["min_train_months"])
        LOG.info("%s: %d months, %d folds", material, len(s), len(origins))
        for model in model_names:
            if (material, model) in done_keys:
                LOG.info("  %-13s already evaluated (resume)", model)
                continue
            rows = []
            t0 = time.time()
            for k, n_train in enumerate(origins):
                train, test = s.iloc[:n_train], s.iloc[n_train:n_train + h]
                fold = len(origins) - k - 1  # 0 = hold-out fold
                t = time.time()
                try:
                    fc = run_model(model, train[["ds", "y"]], h, level, seeds, cfg)
                    status, err = "ok", ""
                except Exception as e:  # recorded, never silently replaced
                    LOG.exception("%s / %s fold %d failed", material, model, fold)
                    fc, status, err = None, "failed", f"{type(e).__name__}: {e}"
                secs = time.time() - t
                for i in range(h):
                    rows.append({
                        "material": material,
                        "model": model,
                        "fold": fold,
                        "origin": train["ds"].iloc[-1],
                        "n_train": n_train,
                        "step": i + 1,
                        "ds": test["ds"].iloc[i],
                        "y": test["y"].iloc[i],
                        "yhat": fc.yhat[i] if fc else np.nan,
                        "lo": fc.lo[i] if fc and fc.lo is not None else np.nan,
                        "hi": fc.hi[i] if fc and fc.hi is not None else np.nan,
                        "info": fc.info if fc else err,
                        "status": status,
                        "seconds": secs,
                    })
            LOG.info("  %-13s %d folds in %.1fs", model, len(origins), time.time() - t0)
            new = pd.DataFrame(rows)
            done = pd.concat([done, new], ignore_index=True) if not done.empty else new
            done.to_csv(pred_path, index=False)
    return done


def fold_metrics(preds: pd.DataFrame, series: dict) -> pd.DataFrame:
    """Accuracy metrics for every (material, model, fold)."""
    out = []
    for (material, model, fold), g in preds.groupby(["material", "model", "fold"]):
        g = g.sort_values("step")
        s = series[material]
        y_train = s["y"].iloc[: int(g["n_train"].iloc[0])]
        row = {"material": material, "model": model, "fold": fold,
               "origin": g["origin"].iloc[0], "seconds": g["seconds"].iloc[0],
               "status": g["status"].iloc[0]}
        if g["status"].iloc[0] == "ok":
            lo = g["lo"] if g["lo"].notna().all() else None
            hi = g["hi"] if g["hi"].notna().all() else None
            row.update(all_metrics(g["y"], g["yhat"], y_train, lo, hi))
        out.append(row)
    return pd.DataFrame(out)
