"""Flask dashboard for procurement planners.

Pick a material, plant, model and horizon; the model is trained on the full
monthly history and the next months are forecast. Alongside the forecast the
page shows how reliable that model was for that material in the
rolling-origin backtest (outputs/tables), so users see accuracy measured on
past data rather than an in-sample number.
"""

import io
import json
import logging

import numpy as np
import pandas as pd
from flask import Flask, Response, render_template, request

from .config import load_config
from .data import build_series, data_end_month, describe_series, load_raw
from .models import MODEL_FAMILY, MODELS, future_index, run_model

LOG = logging.getLogger("forecasting")

CFG = load_config()
RAW, REPORT = load_raw(CFG["data"]["path"], CFG["data"]["sheet"], CFG["data"]["drop_deleted_items"])
END = data_end_month(RAW)
MATERIALS = CFG["series"]["materials"]
SCALE = CFG["series"]["scale_factor"]
LEVEL = CFG["evaluation"]["interval_level"]
TABLES = CFG["paths"]["tables"]

app = Flask(__name__, template_folder="templates")
_CACHE: dict = {}


def _read_table(name):
    p = TABLES / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _plants_for(code):
    return sorted(RAW.loc[RAW["Material"] == code, "Plant"].unique().tolist())


def _best_model(material):
    bt = _read_table("backtest_metrics_by_material.csv")
    bt = bt[bt["material"] == material] if not bt.empty else bt
    return bt.sort_values("MASE")["model"].iloc[0] if not bt.empty else "ETS"


def _forecast(material, plant, model, h):
    key = (material, plant, model, h)
    if key in _CACHE:
        return _CACHE[key]
    s = build_series(RAW, MATERIALS[material], SCALE, plant=plant or None, end=END)
    used, note = model, ""
    if len(s) < h + 4:
        raise ValueError(f"Only {len(s)} months of history - too short to forecast {h} months.")
    try:
        # One seed keeps the dashboard responsive; the thesis results use all seeds.
        fc = run_model(model, s[["ds", "y"]], h, LEVEL, CFG["evaluation"]["seeds"][:1], CFG)
    except Exception as e:
        LOG.exception("dashboard: %s failed", model)
        used, note = "ETS", f"{model} could not be fitted on this series ({type(e).__name__}); ETS fallback shown."
        fc = run_model("ETS", s[["ds", "y"]], h, LEVEL, [1], CFG)
    fut = pd.DataFrame({
        "Month": future_index(s, h),
        "Forecast (millions)": fc.yhat,
        f"Lower {LEVEL}% (millions)": fc.lo if fc.lo is not None else np.nan,
        f"Upper {LEVEL}% (millions)": fc.hi if fc.hi is not None else np.nan,
    })
    fut["Forecast (currency units)"] = fut["Forecast (millions)"] * SCALE
    res = {"series": s, "future": fut, "used": used, "note": note, "info": fc.info,
           "has_interval": fc.lo is not None}
    _CACHE[key] = res
    return res


@app.route("/")
def index():
    ranking = _read_table("overall_ranking.csv")
    summary = _read_table("series_summary.csv")
    return render_template(
        "index.html",
        materials=list(MATERIALS),
        plants={m: _plants_for(c) for m, c in MATERIALS.items()},
        models=list(MODELS),
        families=MODEL_FAMILY,
        default_h=CFG["evaluation"]["horizon"],
        report=REPORT,
        ranking=ranking.round(3).to_dict("records") if not ranking.empty else [],
        summary=summary.to_dict("records") if not summary.empty else [],
    )


@app.route("/forecast")
def forecast():
    material = request.args.get("material", next(iter(MATERIALS)))
    plant = request.args.get("plant", "")
    model = request.args.get("model", "best")
    h = max(1, min(12, request.args.get("horizon", CFG["evaluation"]["horizon"], type=int)))
    if material not in MATERIALS:
        return render_template("message.html", message=f"Unknown material {material}"), 400
    if model == "best":
        model = _best_model(material)
    if model not in MODELS:
        return render_template("message.html", message=f"Unknown model {model}"), 400

    try:
        res = _forecast(material, plant, model, h)
    except Exception as e:
        return render_template("message.html", message=str(e))

    s, fut = res["series"], res["future"]
    traces = [{"x": s["ds"].dt.strftime("%Y-%m-%d").tolist(), "y": s["y"].round(4).tolist(),
               "name": "Actual spend", "type": "scatter", "mode": "lines+markers",
               "line": {"color": "#222", "width": 2}, "marker": {"size": 4}}]
    fx = fut["Month"].dt.strftime("%Y-%m-%d").tolist()
    if res["has_interval"]:
        traces.append({"x": fx + fx[::-1],
                       "y": fut[f"Upper {LEVEL}% (millions)"].tolist() + fut[f"Lower {LEVEL}% (millions)"].tolist()[::-1],
                       "fill": "toself", "fillcolor": "rgba(214,39,40,0.15)", "line": {"width": 0},
                       "mode": "lines",
                       "name": f"{LEVEL}% prediction interval", "type": "scatter", "hoverinfo": "skip"})
    traces.append({"x": [s["ds"].iloc[-1].strftime("%Y-%m-%d")] + fx,
                   "y": [float(s["y"].iloc[-1])] + fut["Forecast (millions)"].round(4).tolist(),
                   "name": f"{res['used']} forecast", "type": "scatter", "mode": "lines+markers",
                   "line": {"color": "#d62728", "width": 3}})

    bt = _read_table("backtest_metrics_by_material.csv")
    board = bt[bt["material"] == material].sort_values("MASE") if not bt.empty else bt
    this = board[board["model"] == res["used"]] if not board.empty else board

    table = fut.copy()
    table["Month"] = table["Month"].dt.strftime("%b %Y")
    return render_template(
        "forecast.html",
        material=material, plant=plant or "All plants", model=res["used"], requested=model,
        family=MODEL_FAMILY.get(res["used"], ""), info=res["info"], note=res["note"], h=h,
        has_interval=res["has_interval"], level=LEVEL,
        overview=describe_series(material, s),
        plot_data=json.dumps(traces),
        rows=table.round(4).to_dict("records"), columns=list(table.columns),
        reliability=this.round(3).to_dict("records")[0] if not this.empty else None,
        board=board.round(3).to_dict("records") if not board.empty else [],
        backtest_on_all_plants=not plant,
        query=request.query_string.decode(),
    )


@app.route("/download")
def download():
    material = request.args.get("material")
    plant = request.args.get("plant", "")
    model = request.args.get("model", "best")
    h = request.args.get("horizon", CFG["evaluation"]["horizon"], type=int)
    if model == "best":
        model = _best_model(material)
    res = _forecast(material, plant, model, h)
    out = res["future"].copy()
    out.insert(0, "Material", material)
    out.insert(1, "Plant", plant or "All")
    out.insert(2, "Model", res["used"])
    out.insert(3, "Model settings", res["info"])
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    name = f"forecast_{material}_{plant or 'all'}_{res['used']}.csv".replace(" ", "_").replace("&", "and")
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={name}"})
