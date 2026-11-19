"""Small local SQLite ledger for reproducible forecast runs."""

from __future__ import annotations

import json
import platform
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

DEFAULT_DB_PATH = Path(__file__).with_name("research_history.db")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def record_experiment(
    symbol: str,
    interval: str,
    requested_model: str,
    result: dict[str, Any],
    feature_names: list[str],
    training_start: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> str:
    """Persist a completed run after evaluation; return its stable run id."""
    experiment_id = str(uuid.uuid4())
    forecast = result["forecast"]
    config = {
        "requested_model": requested_model,
        "selected_model": forecast["model"],
        "features": feature_names,
        "ensemble_weights": forecast["ensemble_weights"],
        "training_start": training_start,
        "validation_end": forecast["validation_end"],
        "validation_points": forecast["validation_points"],
        "random_seed": 17,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    with sqlite3.connect(db_path, timeout=10) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS forecast_experiments (
                experiment_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                horizon INTEGER NOT NULL,
                market_profile TEXT NOT NULL,
                selected_model TEXT NOT NULL,
                reference_price REAL NOT NULL,
                forecast_price REAL NOT NULL,
                return_forecast_pct REAL NOT NULL,
                training_seconds REAL NOT NULL,
                config_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                diagnostics_json TEXT NOT NULL
            )"""
        )
        connection.execute(
            """INSERT INTO forecast_experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                experiment_id,
                datetime.now(timezone.utc).isoformat(),
                symbol,
                interval,
                forecast["horizon"],
                forecast["market_profile"],
                forecast["model"],
                forecast["reference_price"],
                forecast["forecast"],
                forecast["return_forecast_pct"],
                forecast["training_seconds"],
                _json(config),
                _json({"selected": forecast["metrics"], "naive": forecast["naive_metrics"], "comparison": forecast["model_comparison"], "uncertainty": forecast["uncertainty"], "holdout_predictions": forecast["validation_predictions"]}),
                _json({"diagnostics": forecast["diagnostics"], "risk": forecast["risk"]}),
            ),
        )
    return experiment_id


def recent_experiments(symbol: str | None = None, limit: int = 50, db_path: str | Path = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Return recent completed runs, newest first."""
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")
    if not Path(db_path).exists():
        return []
    query = "SELECT experiment_id, created_at, symbol, interval, horizon, market_profile, selected_model, reference_price, forecast_price, return_forecast_pct, training_seconds, config_json, metrics_json, diagnostics_json FROM forecast_experiments"
    parameters: tuple[Any, ...] = ()
    if symbol:
        query += " WHERE symbol = ?"
        parameters = (symbol.upper(),)
    query += " ORDER BY created_at DESC LIMIT ?"
    parameters += (limit,)
    with sqlite3.connect(db_path, timeout=10) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, parameters).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        for key in ("config_json", "metrics_json", "diagnostics_json"):
            item[key.removesuffix("_json")] = json.loads(item.pop(key))
        result.append(item)
    return result
