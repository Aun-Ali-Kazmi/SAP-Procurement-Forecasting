"""All figures used in the thesis, written to outputs/figures as 200-dpi PNGs."""

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_COLORS = {
    "Naive": "#9e9e9e",
    "CrostonSBA": "#bdbdbd",
    "ETS": "#1f77b4",
    "ARIMA": "#17becf",
    "AutoTS": "#2ca02c",
    "MLForecast": "#8c564b",
    "NeuralProphet": "#9467bd",
    "NBEATSx": "#d62728",
    "NHITS": "#ff7f0e",
    "TCN": "#e377c2",
}

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

Y_LABEL = "Monthly spend (millions)"


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    return path


def _slug(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")


# ---------------------------------------------------------------------------
# Chapter 3 - dataset exploration
# ---------------------------------------------------------------------------
def eda_figures(raw: pd.DataFrame, series: dict, out_dir, top_plants: int = 10) -> list:
    paths = []
    end = raw["Document Date"].max()
    plants = raw["Plant"].value_counts().head(top_plants).index.tolist()
    r = raw[raw["Plant"].isin(plants)].copy()

    # Open vs delivered purchase-order lines per plant
    if "Still to be delivered (qty)" in r.columns:
        r["Status"] = np.where(
            pd.to_numeric(r["Still to be delivered (qty)"], errors="coerce").fillna(0) > 0,
            "Open (still to be delivered)", "Fully delivered")
        tab = r.pivot_table(index="Plant", columns="Status", values="Material",
                            aggfunc="count", fill_value=0)
        tab = tab.loc[tab.sum(axis=1).sort_values(ascending=False).index]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        tab.plot.bar(stacked=True, ax=ax, color=["#1f77b4", "#ff7f0e"][: tab.shape[1]])
        ax.set_ylabel("Purchase-order lines")
        ax.set_xlabel("Plant")
        ax.set_title("Open vs delivered purchase-order lines by plant")
        ax.tick_params(axis="x", rotation=0)
        paths.append(_save(fig, out_dir / "fig3_2_open_vs_delivered_by_plant.png"))

        # Ageing of open lines (days since document date, relative to export date)
        open_ = r[r["Status"].str.startswith("Open")].copy()
        if not open_.empty:
            age = (end - open_["Document Date"]).dt.days
            bins = [-1, 30, 60, 90, 120, 365, np.inf]
            labels = ["0-30", "31-60", "61-90", "91-120", "121-365", ">365"]
            open_["Age"] = pd.cut(age, bins=bins, labels=labels)
            tab = open_.pivot_table(index="Age", columns="Plant", values="Material",
                                    aggfunc="count", fill_value=0, observed=False)
            fig, ax = plt.subplots(figsize=(9, 4.5))
            tab.plot.bar(stacked=True, ax=ax, colormap="tab10")
            ax.set_xlabel(f"Age of open line in days (as of {end.date()})")
            ax.set_ylabel("Open purchase-order lines")
            ax.set_title("Ageing of open purchase-order lines")
            ax.tick_params(axis="x", rotation=0)
            ax.legend(title="Plant", ncol=2, fontsize=8)
            paths.append(_save(fig, out_dir / "fig3_3_open_line_ageing.png"))

    # Monthly purchasing activity per plant (heatmap)
    r["Month"] = r["Document Date"].dt.to_period("M")
    heat = r.pivot_table(index="Plant", columns="Month", values="Material",
                         aggfunc="count", fill_value=0)
    heat = heat.loc[heat.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(heat.index)), heat.index)
    ticks = list(range(0, heat.shape[1], 6))
    ax.set_xticks(ticks, [str(heat.columns[i]) for i in ticks], rotation=45, ha="right")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="PO lines")
    ax.set_title("Monthly purchasing activity by plant (number of PO lines)")
    paths.append(_save(fig, out_dir / "fig3_4_monthly_activity_heatmap.png"))

    # Spend by plant
    spend = raw.groupby("Plant")["Net Order Value"].sum().sort_values(ascending=False).head(top_plants)
    fig, ax = plt.subplots(figsize=(9, 4))
    (spend / 1e6).plot.bar(ax=ax, color="#1f77b4")
    ax.set_ylabel("Total net order value (millions)")
    ax.set_title("Total procurement spend by plant")
    ax.tick_params(axis="x", rotation=0)
    paths.append(_save(fig, out_dir / "fig3_5_spend_by_plant.png"))

    # The three modelled series
    fig, axes = plt.subplots(len(series), 1, figsize=(10, 2.6 * len(series)), sharex=True)
    for ax, (name, s) in zip(np.atleast_1d(axes), series.items()):
        ax.bar(s["ds"], s["y"], width=20, color="#1f77b4")
        ax.set_title(f"{name}  ({len(s)} months, {(s['y'] == 0).sum()} with zero spend)", fontsize=10)
        ax.set_ylabel("millions")
    fig.suptitle("Monthly spend of the analysed materials", y=1.0)
    fig.tight_layout()
    paths.append(_save(fig, out_dir / "fig3_6_material_monthly_spend.png"))
    return paths


# ---------------------------------------------------------------------------
# Chapter 7 - results
# ---------------------------------------------------------------------------
def holdout_plot(s: pd.DataFrame, g: pd.DataFrame, material: str, model: str, metrics: dict, path):
    """History, the hold-out months, the forecast and (if real) its interval."""
    g = g.sort_values("ds")
    n_train = int(g["n_train"].iloc[0])
    train, test = s.iloc[:n_train], s.iloc[n_train:n_train + len(g)]
    color = MODEL_COLORS.get(model, "#d62728")

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(train["ds"], train["y"], color="black", lw=1.6, label="Actual (training)")
    ax.plot(test["ds"], test["y"], color="black", lw=1.6, ls="--", marker="o",
            label="Actual (hold-out)")
    # connect last training point to the forecast for readability
    ax.plot([train["ds"].iloc[-1], g["ds"].iloc[0]], [train["y"].iloc[-1], g["yhat"].iloc[0]],
            color=color, lw=1, ls=":")
    ax.plot(g["ds"], g["yhat"], color=color, lw=2.2, marker="o", label=f"{model} forecast")
    if g["lo"].notna().all():
        ax.fill_between(g["ds"], g["lo"], g["hi"], color=color, alpha=0.18,
                        label="80% prediction interval")
    ax.axvline(train["ds"].iloc[-1] + pd.Timedelta(days=15), color="grey", lw=1, ls="--")
    ax.set_title(f"{model} - {material}   (hold-out MAE {metrics['MAE']:.3f}, "
                 f"MASE {metrics['MASE']:.2f})")
    ax.set_ylabel(Y_LABEL)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=8, frameon=False)
    return _save(fig, path)


def holdout_overlay(s, g_all: pd.DataFrame, material, path, last_months: int = 24):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    n_train = int(g_all["n_train"].iloc[0])
    hist = s.iloc[max(0, n_train - last_months):n_train + g_all["step"].max()]
    ax.plot(hist["ds"], hist["y"], color="black", lw=2, marker=".", label="Actual")
    for model, g in g_all.groupby("model", sort=False):
        g = g.sort_values("ds")
        ax.plot(g["ds"], g["yhat"], lw=1.6, marker="o", ms=3,
                color=MODEL_COLORS.get(model), label=model)
    ax.axvline(s["ds"].iloc[n_train - 1] + pd.Timedelta(days=15), color="grey", ls="--", lw=1)
    ax.set_title(f"All models on the hold-out months - {material}")
    ax.set_ylabel(Y_LABEL)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=6, fontsize=8, frameon=False)
    return _save(fig, path)


def metric_bars(table: pd.DataFrame, metric: str, title: str, path, order):
    """Grouped bar chart: models (x) x materials (bars)."""
    piv = table.pivot(index="model", columns="material", values=metric).reindex(order)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    piv.plot.bar(ax=ax, width=0.8, colormap="tab10")
    ax.set_ylabel(metric)
    ax.set_xlabel("")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Material", fontsize=8)
    return _save(fig, path)


def ranking_bar(rank: pd.DataFrame, path):
    r = rank.sort_values("mean_MASE")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(r["model"], r["mean_MASE"], color=[MODEL_COLORS.get(m) for m in r["model"]])
    ax.axvline(1.0, color="grey", ls="--", lw=1)
    ax.invert_yaxis()
    for i, v in enumerate(r["mean_MASE"]):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    ax.set_xlabel("Mean MASE over all materials and backtest folds (lower is better)")
    ax.set_title("Overall model ranking (rolling-origin backtest)")
    return _save(fig, path)


def mase_boxplot(fm: pd.DataFrame, order, path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    data = [fm.loc[fm["model"] == m, "MASE"].dropna() for m in order]
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showfliers=True)
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(MODEL_COLORS.get(m))
        patch.set_alpha(0.6)
    ax.axhline(1.0, color="grey", ls="--", lw=1)
    ax.set_ylabel("MASE per fold")
    ax.set_title("Stability of accuracy across backtest folds and materials")
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, path)


def future_plot(s, fut: pd.DataFrame, material, model, path, last_months: int = 36):
    color = MODEL_COLORS.get(model, "#d62728")
    hist = s.tail(last_months)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(hist["ds"], hist["y"], color="black", lw=1.6, marker=".", label="Actual")
    ax.plot(fut["ds"], fut["yhat"], color=color, lw=2.2, marker="o", label=f"{model} forecast")
    if fut["lo"].notna().all():
        ax.fill_between(fut["ds"], fut["lo"], fut["hi"], color=color, alpha=0.18,
                        label="80% prediction interval")
    ax.set_title(f"Next-{len(fut)}-month spend forecast - {material} (best backtest model: {model})")
    ax.set_ylabel(Y_LABEL)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=8, frameon=False)
    return _save(fig, path)


slug = _slug
