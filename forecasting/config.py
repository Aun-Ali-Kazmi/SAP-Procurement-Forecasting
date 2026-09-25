"""Load config.yaml and resolve all paths relative to the project root."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["data"]["path"] = str(PROJECT_ROOT / cfg["data"]["path"])
    out = PROJECT_ROOT / cfg["output"]["dir"]
    cfg["paths"] = {
        "out": out,
        "figures": out / "figures",
        "tables": out / "tables",
        "forecasts": out / "forecasts",
        "logs": out / "logs",
        "processed": PROJECT_ROOT / "data" / "processed",
    }
    for p in cfg["paths"].values():
        p.mkdir(parents=True, exist_ok=True)
    return cfg
