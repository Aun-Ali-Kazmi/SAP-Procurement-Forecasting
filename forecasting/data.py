"""Ingestion, validation and monthly aggregation of SAP purchase-order exports.

The SAP export contains one row per purchase-order line. Forecasting models
need a regular time series, so each material is aggregated to total monthly
spend (sum of Net Order Value by Document Date month). Months with no
purchasing are recorded as zero spend: in procurement data an empty month is
information, not a missing value. No synthetic history is ever generated.
"""

import logging

import numpy as np
import pandas as pd

LOG = logging.getLogger("forecasting")

REQUIRED_COLUMNS = [
    "Plant",
    "Material",
    "MATERIAL DETAILS",
    "Document Date",
    "Net Order Value",
]


def load_raw(path: str, sheet: str | None = None, drop_deleted: bool = True):
    """Read the SAP export, validate the schema and clean types.

    Returns the cleaned DataFrame and a dict describing every row that was
    removed and why (written to outputs/tables/data_validation.csv).
    """
    df = pd.read_excel(path, sheet_name=sheet or 0)
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"SAP export is missing required columns: {missing}")

    report = {"rows_read": len(df)}

    empty = df["Material"].isna() & df["Document Date"].isna()
    report["rows_empty"] = int(empty.sum())
    df = df[~empty]

    # Excel turns material numbers into floats (140003543.0) -> normalise.
    df["Material"] = (
        df["Material"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    )
    df["Plant"] = df["Plant"].astype(str).str.strip()
    df["MATERIAL DETAILS"] = df["MATERIAL DETAILS"].astype(str).str.strip()
    df["Document Date"] = pd.to_datetime(df["Document Date"], errors="coerce")
    df["Net Order Value"] = pd.to_numeric(df["Net Order Value"], errors="coerce")

    if drop_deleted and "Deletion indicator" in df.columns:
        deleted = df["Deletion indicator"].notna() & (
            df["Deletion indicator"].astype(str).str.strip() != ""
        )
        report["rows_deleted_in_sap"] = int(deleted.sum())
        df = df[~deleted]

    bad_date = df["Document Date"].isna()
    report["rows_invalid_date"] = int(bad_date.sum())
    df = df[~bad_date]

    bad_value = df["Net Order Value"].isna() | (df["Net Order Value"] < 0)
    report["rows_invalid_or_negative_value"] = int(bad_value.sum())
    df = df[~bad_value]

    dup = df.duplicated()
    report["rows_exact_duplicates"] = int(dup.sum())
    df = df[~dup]

    report["rows_kept"] = len(df)
    report["date_min"] = df["Document Date"].min().date().isoformat()
    report["date_max"] = df["Document Date"].max().date().isoformat()
    if "Currency" in df.columns:
        report["currencies"] = ", ".join(sorted(df["Currency"].dropna().astype(str).unique()))

    for k, v in report.items():
        LOG.info("data validation: %s = %s", k, v)
    return df.reset_index(drop=True), report


def data_end_month(df: pd.DataFrame, min_days: int = 25) -> pd.Timestamp:
    """Last *complete* month in the export (the common series end).

    If the export stops early in a month (e.g. on the 10th) that month's
    spend is only partly recorded and would look like a sudden drop, so it
    is excluded.
    """
    last = df["Document Date"].max()
    month = last.to_period("M")
    if last.day < min_days:
        LOG.info("export ends on %s: partial month %s excluded", last.date(), month)
        month -= 1
    return month.to_timestamp()


def build_series(
    df: pd.DataFrame,
    material_code: str,
    scale_factor: float,
    plant: str | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Monthly spend series for one material (optionally one plant).

    Columns: ds (month start), spend (original currency), y (spend / scale).
    The series runs from the first month with a purchase to `end` (default:
    the last month in the export), with zero-spend months filled in.
    """
    sub = df[df["Material"] == str(material_code)]
    if plant:
        sub = sub[sub["Plant"] == plant]
    if sub.empty:
        return pd.DataFrame(columns=["ds", "spend", "y"])

    monthly = sub.groupby(sub["Document Date"].dt.to_period("M"))["Net Order Value"].sum()
    monthly.index = monthly.index.to_timestamp()

    end = end if end is not None else data_end_month(df)
    idx = pd.date_range(monthly.index.min(), end, freq="MS")
    monthly = monthly.reindex(idx, fill_value=0.0)

    out = pd.DataFrame({"ds": idx, "spend": monthly.values.astype(float)})
    out["y"] = out["spend"] / scale_factor
    return out


def describe_series(name: str, s: pd.DataFrame) -> dict:
    y = s["y"].to_numpy()
    nz = y[y > 0]
    # Average inter-demand interval and squared coefficient of variation are
    # the standard Syntetos-Boylan measures of intermittency.
    adi = len(y) / max(len(nz), 1)
    cv2 = float((nz.std() / nz.mean()) ** 2) if len(nz) > 1 else np.nan
    if adi < 1.32 and cv2 < 0.49:
        demand_class = "Smooth"
    elif adi >= 1.32 and cv2 < 0.49:
        demand_class = "Intermittent"
    elif adi < 1.32:
        demand_class = "Erratic"
    else:
        demand_class = "Lumpy"
    return {
        "material": name,
        "first_month": s["ds"].min().date().isoformat(),
        "last_month": s["ds"].max().date().isoformat(),
        "months": len(s),
        "zero_months": int((y == 0).sum()),
        "mean_spend_m": round(float(y.mean()), 4),
        "max_spend_m": round(float(y.max()), 4),
        "ADI": round(adi, 3),
        "CV2": round(cv2, 3) if not np.isnan(cv2) else None,
        "demand_class": demand_class,
    }
