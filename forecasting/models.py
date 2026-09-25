"""Forecasting models behind one common interface.

Every model is a function

    fn(train: DataFrame[ds, y], h: int, level: int, seed: int, cfg: dict) -> Forecast

that fits on the training months only and returns `h` point forecasts plus,
*only when the model genuinely produces one*, a prediction interval. Models
that have no native uncertainty estimate return lo/hi = None; no interval is
ever fabricated. Forecasts of spend are clipped at zero.
"""

import contextlib
import io
import logging
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOG = logging.getLogger("forecasting")
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Silence the very chatty training loggers of the deep-learning libraries.
for _name in ("pytorch_lightning", "lightning", "lightning.pytorch", "lightning_fabric",
              "neuralprophet", "NP", "darts", "cmdstanpy", "prophet"):
    logging.getLogger(_name).setLevel(logging.ERROR)


@dataclass
class Forecast:
    yhat: np.ndarray
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    info: str = ""

    def clipped(self) -> "Forecast":
        """Non-negative spend, and bounds re-ordered so lo <= yhat <= hi.

        Quantile models learn the median and the bounds with separate outputs
        which can cross on small data; sorting them is the standard remedy.
        """
        c = lambda a: None if a is None else np.clip(np.asarray(a, float), 0, None)
        yhat, lo, hi = c(self.yhat), c(self.lo), c(self.hi)
        if lo is not None and hi is not None:
            lo, hi = np.minimum.reduce([lo, hi, yhat]), np.maximum.reduce([lo, hi, yhat])
        return Forecast(yhat, lo, hi, self.info)


def _future_dates(train: pd.DataFrame, h: int) -> pd.DatetimeIndex:
    return pd.date_range(train["ds"].iloc[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")


def _lookback(n_train: int, h: int, cap: int) -> int:
    """Largest look-back window the training series can support."""
    return int(max(h, min(cap, n_train - h)))


def _trainer_kwargs() -> dict:
    return dict(
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def naive(train, h, level, seed, cfg):
    """Last observed month repeated (the minimum any model must beat)."""
    return Forecast(np.repeat(train["y"].iloc[-1], h), info="last value")


def croston_sba(train, h, level, seed, cfg, alpha: float = 0.1):
    """Croston's method with the Syntetos-Boylan bias correction.

    The standard benchmark for intermittent demand (many zero months):
    demand sizes and inter-demand intervals are smoothed separately.
    """
    y = train["y"].to_numpy(float)
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0:
        return Forecast(np.zeros(h), info="no demand")
    z, p, q = y[nz[0]], float(nz[0] + 1), 1.0
    for t in range(nz[0] + 1, len(y)):
        if y[t] > 0:
            z = z + alpha * (y[t] - z)
            p = p + alpha * (q - p)
            q = 1.0
        else:
            q += 1.0
    return Forecast(np.repeat((1 - alpha / 2) * z / p, h), info=f"alpha={alpha}")


# ---------------------------------------------------------------------------
# Classical statistical models
# ---------------------------------------------------------------------------
def ets(train, h, level, seed, cfg):
    """Exponential smoothing (state-space ETS), configuration chosen by AICc."""
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    y = pd.Series(train["y"].to_numpy(float), index=pd.DatetimeIndex(train["ds"], freq="MS"))
    candidates = [dict(trend=None, damped_trend=False)]
    if len(y) >= 10:
        candidates.append(dict(trend="add", damped_trend=True))
    if len(y) >= 36:
        candidates += [dict(trend=None, damped_trend=False, seasonal="add", seasonal_periods=12)]

    best, best_spec = None, None
    for spec in candidates:
        try:
            fit = ETSModel(y, error="add", **spec).fit(disp=False)
            if best is None or fit.aicc < best.aicc:
                best, best_spec = fit, spec
        except Exception:
            continue
    if best is None:
        raise RuntimeError("no ETS configuration could be fitted")

    pred = best.get_prediction(start=len(y), end=len(y) + h - 1)
    frame = pred.summary_frame(alpha=1 - level / 100)
    label = "A" + ("Ad" if best_spec.get("damped_trend") else "N") + ("A" if best_spec.get("seasonal") else "N")
    return Forecast(frame["mean"].to_numpy(), frame["pi_lower"].to_numpy(),
                    frame["pi_upper"].to_numpy(), info=f"ETS({label})")


def arima(train, h, level, seed, cfg):
    """ARIMA(p,d,q) with the order selected by AIC over a small grid."""
    from statsmodels.tsa.arima.model import ARIMA

    y = pd.Series(train["y"].to_numpy(float), index=pd.DatetimeIndex(train["ds"], freq="MS"))
    best, best_order = None, None
    for d in (0, 1):
        for p in (0, 1, 2):
            for q in (0, 1, 2):
                if p + q > max(1, len(y) // 6):
                    continue
                try:
                    fit = ARIMA(y, order=(p, d, q)).fit()
                    if np.isfinite(fit.aic) and (best is None or fit.aic < best.aic):
                        best, best_order = fit, (p, d, q)
                except Exception:
                    continue
    if best is None:
        raise RuntimeError("no ARIMA order could be fitted")
    fc = best.get_forecast(h)
    ci = fc.conf_int(alpha=1 - level / 100)
    return Forecast(fc.predicted_mean.to_numpy(), ci.iloc[:, 0].to_numpy(),
                    ci.iloc[:, 1].to_numpy(), info=f"ARIMA{best_order}")


# ---------------------------------------------------------------------------
# Automated / machine-learning models
# ---------------------------------------------------------------------------
def autots(train, h, level, seed, cfg):
    """AutoTS genetic search over model families with a simple ensemble."""
    from autots import AutoTS

    wide = pd.DataFrame({"y": train["y"].to_numpy(float)}, index=pd.DatetimeIndex(train["ds"]))
    n_val = 2 if len(wide) >= 5 * h + 6 else 1
    model = AutoTS(
        forecast_length=h,
        frequency="MS",
        prediction_interval=level / 100,
        ensemble="simple",
        model_list="fast",
        transformer_list="fast",
        max_generations=5,
        num_validations=n_val,
        validation_method="backwards",
        min_allowed_train_percent=0.3,
        random_seed=seed,
        n_jobs=1,
        verbose=0,
    )
    # AutoTS prints every candidate model it tries; keep the log readable.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = model.fit(wide)
        pred = model.predict()
    return Forecast(pred.forecast["y"].to_numpy(), pred.lower_forecast["y"].to_numpy(),
                    pred.upper_forecast["y"].to_numpy(), info=str(model.best_model_name))


def mlforecast(train, h, level, seed, cfg):
    """LightGBM on lag + calendar features, recursive multi-step (Nixtla MLForecast).

    Prediction intervals come from conformal prediction on past windows and
    are only produced when the series is long enough to calibrate them.
    """
    import lightgbm as lgb
    from mlforecast import MLForecast
    from mlforecast.utils import PredictionIntervals

    n = len(train)
    lags = [l for l in (1, 2, 3, 6, 12) if l <= n - h - 3]
    df = pd.DataFrame({"unique_id": "s", "ds": train["ds"], "y": train["y"].to_numpy(float)})
    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=8,
                              min_child_samples=3, random_state=seed, verbosity=-1)
    fc = MLForecast(models={"lgbm": model}, freq="MS", lags=lags, date_features=["month"])

    intervals = None
    if n - max(lags) >= 3 * h + 6:
        intervals = PredictionIntervals(n_windows=2, h=h)
    fc.fit(df, prediction_intervals=intervals)
    out = fc.predict(h, level=[level] if intervals else None)
    lo = out[f"lgbm-lo-{level}"].to_numpy() if intervals else None
    hi = out[f"lgbm-hi-{level}"].to_numpy() if intervals else None
    return Forecast(out["lgbm"].to_numpy(), lo, hi, info=f"lags={lags}")


# ---------------------------------------------------------------------------
# Deep-learning models
# ---------------------------------------------------------------------------
def neuralprophet(train, h, level, seed, cfg):
    """NeuralProphet (trend + seasonality + neural components), quantile output."""
    from neuralprophet import NeuralProphet, set_log_level, set_random_seed

    set_log_level("ERROR")
    set_random_seed(seed)
    q = (1 - level / 100) / 2
    n = len(train)
    m = NeuralProphet(
        n_changepoints=max(1, min(10, n // 4)),
        yearly_seasonality=n >= 24,
        weekly_seasonality=False,
        daily_seasonality=False,
        epochs=cfg["models"]["deep"]["neuralprophet_epochs"],
        learning_rate=0.05,
        loss_func="Huber",
        quantiles=[q, 1 - q],
        trainer_config=_trainer_kwargs(),
    )
    df = train[["ds", "y"]].copy()
    m.fit(df, freq="MS", progress=None)
    future = m.make_future_dataframe(df, periods=h)
    pred = m.predict(future).tail(h)
    lo_col = [c for c in pred.columns if c.startswith("yhat1 ") and f"{q * 100:.1f}" in c][0]
    hi_col = [c for c in pred.columns if c.startswith("yhat1 ") and f"{(1 - q) * 100:.1f}" in c][0]
    return Forecast(pred["yhat1"].to_numpy(), pred[lo_col].to_numpy(), pred[hi_col].to_numpy())


def _neuralforecast(model_cls, train, h, level, seed, cfg, **kwargs):
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MQLoss

    deep = cfg["models"]["deep"]
    input_size = _lookback(len(train), h, deep["max_input_size"])
    model = model_cls(
        h=h,
        input_size=input_size,
        loss=MQLoss(level=[level]),
        max_steps=deep["max_steps"],
        learning_rate=1e-3,
        scaler_type="standard",
        start_padding_enabled=True,
        random_seed=seed,
        **kwargs,
        **_trainer_kwargs(),
    )
    df = pd.DataFrame({"unique_id": "s", "ds": train["ds"], "y": train["y"].to_numpy(float)})
    nf = NeuralForecast(models=[model], freq="MS")
    nf.fit(df)
    out = nf.predict()
    name = model_cls.__name__
    return Forecast(out[f"{name}-median"].to_numpy(), out[f"{name}-lo-{level}"].to_numpy(),
                    out[f"{name}-hi-{level}"].to_numpy(), info=f"input_size={input_size}")


def nbeatsx(train, h, level, seed, cfg):
    """N-BEATSx: deep residual stacks of basis-expansion blocks (Olivares et al., 2023)."""
    from neuralforecast.models import NBEATSx

    return _neuralforecast(NBEATSx, train, h, level, seed, cfg,
                           mlp_units=[[128, 128], [128, 128]])


def nhits(train, h, level, seed, cfg):
    """N-HiTS: hierarchical interpolation with multi-rate sampling (Challu et al., 2023)."""
    from neuralforecast.models import NHITS

    return _neuralforecast(NHITS, train, h, level, seed, cfg,
                           mlp_units=[[128, 128], [128, 128]],
                           n_pool_kernel_size=[2, 1, 1], n_freq_downsample=[2, 1, 1])


def tcn(train, h, level, seed, cfg):
    """Temporal Convolutional Network (Darts) with quantile-regression output."""
    from darts import TimeSeries
    from darts.dataprocessing.transformers import Scaler
    from darts.models import TCNModel
    from darts.utils.likelihood_models import QuantileRegression

    q = (1 - level / 100) / 2
    series = TimeSeries.from_times_and_values(pd.DatetimeIndex(train["ds"], freq="MS"),
                                              train["y"].to_numpy(float))
    scaler = Scaler()  # fitted on the training months only
    series_s = scaler.fit_transform(series)

    input_len = max(h + 1, min(cfg["models"]["deep"]["max_input_size"], len(train) - h))
    model = TCNModel(
        input_chunk_length=input_len,
        output_chunk_length=h,
        kernel_size=2,
        num_filters=8,
        dropout=0.1,
        n_epochs=cfg["models"]["deep"]["tcn_epochs"],
        likelihood=QuantileRegression(quantiles=[q, 0.5, 1 - q]),
        random_state=seed,
        pl_trainer_kwargs=_trainer_kwargs(),
    )
    model.fit(series_s, verbose=False)
    pred = scaler.inverse_transform(model.predict(n=h, num_samples=300, verbose=False))
    samples = pred.all_values()[:, 0, :]  # (h, num_samples)
    return Forecast(np.median(samples, axis=1), np.quantile(samples, q, axis=1),
                    np.quantile(samples, 1 - q, axis=1), info=f"input_chunk={input_len}")


MODELS = {
    "Naive": naive,
    "CrostonSBA": croston_sba,
    "ETS": ets,
    "ARIMA": arima,
    "AutoTS": autots,
    "MLForecast": mlforecast,
    "NeuralProphet": neuralprophet,
    "NBEATSx": nbeatsx,
    "NHITS": nhits,
    "TCN": tcn,
}

# Models whose result depends on the random seed; they are trained once per
# seed in config.evaluation.seeds and the forecasts are averaged.
STOCHASTIC = {"NeuralProphet", "NBEATSx", "NHITS", "TCN"}

MODEL_FAMILY = {
    "Naive": "Benchmark",
    "CrostonSBA": "Benchmark (intermittent)",
    "ETS": "Statistical",
    "ARIMA": "Statistical",
    "AutoTS": "Automated ML",
    "MLForecast": "Machine learning",
    "NeuralProphet": "Deep learning",
    "NBEATSx": "Deep learning",
    "NHITS": "Deep learning",
    "TCN": "Deep learning",
}


def run_model(name: str, train: pd.DataFrame, h: int, level: int, seeds, cfg) -> Forecast:
    """Fit `name` on `train` and forecast `h` months (seed-averaged if stochastic)."""
    fn = MODELS[name]
    use_seeds = list(seeds) if name in STOCHASTIC else [seeds[0]]
    runs = [fn(train, h, level, s, cfg) for s in use_seeds]
    avg = lambda arrs: None if any(a is None for a in arrs) else np.mean(np.vstack(arrs), axis=0)
    fc = Forecast(avg([r.yhat for r in runs]), avg([r.lo for r in runs]),
                  avg([r.hi for r in runs]), info=runs[0].info).clipped()
    # A zero-width band is not an uncertainty estimate (e.g. an undertrained
    # quantile head); report "no interval" rather than a misleading one.
    if fc.lo is not None and np.allclose(fc.hi - fc.lo, 0, atol=1e-9):
        fc = Forecast(fc.yhat, None, None, (fc.info + "; degenerate interval dropped").strip("; "))
    return fc


def future_index(train: pd.DataFrame, h: int) -> pd.DatetimeIndex:
    return _future_dates(train, h)
