"""End-to-end pipeline: SAP export -> clean monthly series -> 10 models ->
rolling-origin backtest -> tables and figures for the thesis.

    python run_pipeline.py            # full run (resumes if interrupted)
    python run_pipeline.py --fresh    # discard previous predictions and rerun
    python run_pipeline.py --quick    # fast smoke test (1 seed, fewer steps)
"""

import argparse
import warnings
import hashlib
import json
import logging
import platform
import sys
import time
from datetime import datetime
from importlib import metadata

import numpy as np
import pandas as pd

from forecasting import plots
from forecasting.config import load_config
from forecasting.data import build_series, data_end_month, describe_series, load_raw
from forecasting.evaluation import backtest, fold_metrics
from forecasting.models import MODEL_FAMILY, future_index, run_model

LOG = logging.getLogger("forecasting")
warnings.filterwarnings("ignore")


def setup_logging(log_dir):
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")):
        h.setFormatter(fmt)
        LOG.addHandler(h)


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "-" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write_metadata(cfg, args, report, started):
    pkgs = ["numpy", "pandas", "statsmodels", "scikit-learn", "lightgbm", "mlforecast",
            "autots", "neuralprophet", "neuralforecast", "darts", "torch"]
    versions = {}
    for p in pkgs:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    with open(cfg["data"]["path"], "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    meta = {
        "run_started": started,
        "run_finished": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "data_file_sha256": sha,
        "data_validation": report,
        "arguments": vars(args),
        "config": {k: v for k, v in cfg.items() if k != "paths"},
    }
    path = cfg["paths"]["out"] / "run_metadata.json"
    path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--fresh", action="store_true", help="ignore previously saved predictions")
    ap.add_argument("--quick", action="store_true", help="1 seed, fewer training steps (smoke test)")
    ap.add_argument("--models", nargs="*", help="subset of models to run")
    ap.add_argument("--skip-future", action="store_true", help="skip the next-months forecast")
    args = ap.parse_args()

    started = datetime.now().isoformat(timespec="seconds")
    cfg = load_config(args.config)
    P = cfg["paths"]
    if args.quick:
        cfg["evaluation"]["seeds"] = cfg["evaluation"]["seeds"][:1]
        cfg["evaluation"]["max_origins"] = 2
        cfg["models"]["deep"].update(max_steps=50, tcn_epochs=30, neuralprophet_epochs=30)
        base = P["out"] / "quick"
        P["out"] = base
        for k in ("figures", "tables", "forecasts"):
            P[k] = base / k
            P[k].mkdir(parents=True, exist_ok=True)
    setup_logging(P["logs"])
    fig_dir, tab_dir = P["figures"], P["tables"]
    model_names = args.models or cfg["models"]["enabled"]
    h = cfg["evaluation"]["horizon"]
    t_start = time.time()

    # 1. Load and validate --------------------------------------------------
    LOG.info("Step 1/6  loading %s", cfg["data"]["path"])
    raw, report = load_raw(cfg["data"]["path"], cfg["data"]["sheet"], cfg["data"]["drop_deleted_items"])
    pd.DataFrame([report]).T.rename(columns={0: "value"}).to_csv(tab_dir / "data_validation.csv")

    # 2. Monthly series -----------------------------------------------------
    LOG.info("Step 2/6  building monthly series")
    end = data_end_month(raw)
    series = {}
    for name, code in cfg["series"]["materials"].items():
        s = build_series(raw, code, cfg["series"]["scale_factor"], end=end)
        if s.empty:
            LOG.warning("no purchases found for %s (%s) - skipped", name, code)
            continue
        s.to_csv(P["processed"] / f"series_{plots.slug(name)}.csv", index=False)
        series[name] = s
    summary = pd.DataFrame([describe_series(n, s) for n, s in series.items()])
    summary.to_csv(tab_dir / "series_summary.csv", index=False)
    LOG.info("\n%s", summary.to_string(index=False))

    # 3. Exploratory figures ----------------------------------------------
    LOG.info("Step 3/6  exploratory figures")
    plots.eda_figures(raw, series, fig_dir)

    # 4. Rolling-origin backtest ------------------------------------------
    LOG.info("Step 4/6  rolling-origin backtest (%s)", ", ".join(model_names))
    pred_path = P["forecasts"] / "backtest_predictions.csv"
    if args.fresh and pred_path.exists():
        pred_path.unlink()
    preds = backtest(series, model_names, cfg, pred_path)
    preds = preds[preds["model"].isin(model_names) & preds["material"].isin(series)]
    fm = fold_metrics(preds, series)
    fm.to_csv(tab_dir / "backtest_fold_metrics.csv", index=False)

    failed = fm[fm["status"] != "ok"]
    if not failed.empty:
        LOG.warning("%d model fits failed - see backtest_fold_metrics.csv", len(failed))
    order = [m for m in model_names if m in set(fm["model"])]

    # 5. Tables -------------------------------------------------------------
    LOG.info("Step 5/6  tables and figures")
    ok = fm[fm["status"] == "ok"].copy()
    ok["rank"] = ok.groupby(["material", "fold"])["MAE"].rank(method="average")

    # Hold-out (fold 0) = the thesis' Table 7
    hold = ok[ok["fold"] == 0]
    hold.to_csv(tab_dir / "holdout_metrics_long.csv", index=False)
    t7 = hold.pivot(index="model", columns="material", values="MAE").reindex(order)
    t7["Mean MASE"] = hold.groupby("model")["MASE"].mean().reindex(order)
    t7["Mean rank"] = hold.groupby("model")["rank"].mean().reindex(order)
    t7 = t7.sort_values("Mean MASE").reset_index()
    t7.insert(1, "Family", t7["model"].map(MODEL_FAMILY))
    t7.to_csv(tab_dir / "table_holdout_MAE.csv", index=False)

    # Backtest averages per material and overall ranking
    bt = ok.groupby(["material", "model"]).agg(
        MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), MAPE=("MAPE", "mean"),
        sMAPE=("sMAPE", "mean"), MASE=("MASE", "mean"), Bias=("Bias", "mean"),
        PICP=("PICP", "mean"), folds=("fold", "count"), sec_per_fit=("seconds", "mean"),
    ).reset_index()
    bt.to_csv(tab_dir / "backtest_metrics_by_material.csv", index=False)

    rank = ok.groupby("model").agg(
        mean_MASE=("MASE", "mean"), median_MASE=("MASE", "median"), mean_rank=("rank", "mean"),
        wins=("rank", lambda r: int((r == 1).sum())), PICP_80=("PICP", "mean"),
        sec_per_fit=("seconds", "mean"), fits=("fold", "count"),
    ).reset_index()
    beats_naive = []
    naive_mae = ok[ok["model"] == "Naive"].set_index(["material", "fold"])["MAE"]
    for m in rank["model"]:
        mm = ok[ok["model"] == m].set_index(["material", "fold"])["MAE"]
        common = mm.index.intersection(naive_mae.index)
        beats_naive.append(float((mm[common] < naive_mae[common]).mean()) if len(common) else np.nan)
    rank["share_folds_beating_naive"] = beats_naive
    rank.insert(1, "Family", rank["model"].map(MODEL_FAMILY))
    rank = rank.sort_values("mean_MASE").reset_index(drop=True)
    rank.to_csv(tab_dir / "overall_ranking.csv", index=False)

    bt_mae = bt.pivot(index="model", columns="material", values="MAE").reindex(order).reset_index()
    md = [
        "# Results tables (generated by run_pipeline.py)\n",
        "## Series summary\n", to_markdown(summary), "",
        f"## Table 7 - hold-out MAE (last {h} months, spend in millions)\n", to_markdown(t7), "",
        "## Rolling-origin backtest - mean MAE per material\n", to_markdown(bt_mae), "",
        "## Overall ranking (all materials and folds)\n", to_markdown(rank), "",
    ]
    (tab_dir / "results_tables.md").write_text("\n".join(md), encoding="utf-8")

    # 6. Figures ------------------------------------------------------------
    hold_preds = preds[(preds["fold"] == 0) & (preds["status"] == "ok")]
    for material, s in series.items():
        g_mat = hold_preds[hold_preds["material"] == material]
        for model in order:
            g = g_mat[g_mat["model"] == model]
            if g.empty:
                continue
            m = hold[(hold["material"] == material) & (hold["model"] == model)].iloc[0].to_dict()
            plots.holdout_plot(s, g, material, model, m,
                               fig_dir / f"fig7_holdout_{plots.slug(material)}_{model}.png")
        plots.holdout_overlay(s, g_mat, material, fig_dir / f"fig7_holdout_all_models_{plots.slug(material)}.png")
    plots.metric_bars(hold, "MAE", f"Hold-out MAE by model and material (last {h} months)",
                      fig_dir / "fig7_holdout_MAE_by_model.png", order)
    plots.metric_bars(bt, "MAE", "Rolling-origin backtest: mean MAE by model and material",
                      fig_dir / "fig7_backtest_MAE_by_model.png", order)
    plots.ranking_bar(rank, fig_dir / "fig7_overall_ranking_MASE.png")
    plots.mase_boxplot(ok, order, fig_dir / "fig7_backtest_MASE_boxplot.png")

    # 7. Forecast of the next months with every model -----------------------
    if not args.skip_future:
        LOG.info("Step 6/6  forecasting the next %d months", h)
        rows = []
        for material, s in series.items():
            dates = future_index(s, h)
            for model in order:
                try:
                    fc = run_model(model, s[["ds", "y"]], h, cfg["evaluation"]["interval_level"],
                                   cfg["evaluation"]["seeds"], cfg)
                except Exception as e:
                    LOG.warning("future forecast %s / %s failed: %s", material, model, e)
                    continue
                for i, d in enumerate(dates):
                    rows.append({"material": material, "model": model, "ds": d,
                                 "yhat": fc.yhat[i],
                                 "lo": fc.lo[i] if fc.lo is not None else np.nan,
                                 "hi": fc.hi[i] if fc.hi is not None else np.nan,
                                 "yhat_spend": fc.yhat[i] * cfg["series"]["scale_factor"]})
            fut = pd.DataFrame(rows)
            best = bt[bt["material"] == material].sort_values("MASE")["model"].iloc[0]
            fb = fut[(fut["material"] == material) & (fut["model"] == best)]
            if not fb.empty:
                plots.future_plot(s, fb, material, best, fig_dir / f"fig7_future_{plots.slug(material)}.png")
        pd.DataFrame(rows).to_csv(tab_dir / "future_forecasts.csv", index=False)

    meta = write_metadata(cfg, args, report, started)
    LOG.info("\n%s", t7.to_string(index=False))
    LOG.info("\n%s", rank.to_string(index=False))
    LOG.info("Done in %.1f min. Tables: %s  Figures: %s  Metadata: %s",
             (time.time() - t_start) / 60, tab_dir, fig_dir, meta)


if __name__ == "__main__":
    main()
