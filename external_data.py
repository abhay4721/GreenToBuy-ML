"""Point-in-time macro features from no-key public financial data sources."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import requests

FRANKFURTER_API = "https://api.frankfurter.dev/v2/rates"


@lru_cache(maxsize=12)
def _fetch_usd_inr_history(start: str, end: str) -> tuple[tuple[str, float], ...]:
    response = requests.get(
        FRANKFURTER_API,
        params={"from": start, "to": end, "base": "USD", "quotes": "INR"},
        timeout=(3, 10),
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Frankfurter returned an unexpected response format.")
    rows = []
    for item in payload:
        try:
            if item["base"] != "USD" or item["quote"] != "INR":
                continue
            rate = float(item["rate"])
            if np.isfinite(rate) and rate > 0:
                rows.append((str(item["date"]), rate))
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(sorted(set(rows)))


def indian_macro_features(index: pd.Index) -> tuple[pd.DataFrame | None, dict]:
    """Return USD/INR reference-rate features strictly lagged by one date.

    Frankfurter rates are reference rates, not intraday executable quotes. The
    one-calendar-day availability lag avoids using a rate on its observation date.
    Missing or stale context is reported and callers can continue without it.
    """
    if len(index) == 0:
        return None, {"available": False, "reason": "No market timestamps."}
    bars = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    start = (bars.min().normalize() - pd.Timedelta(days=7)).date().isoformat()
    end = bars.max().date().isoformat()
    try:
        rows = _fetch_usd_inr_history(start, end)
        if not rows:
            return None, {"available": False, "reason": "No USD/INR reference rates were returned."}

        rates = pd.DataFrame(rows, columns=["rate_date", "usd_inr_rate"])
        rates["rate_date"] = pd.to_datetime(rates["rate_date"], utc=True).dt.normalize()
        rates = rates.drop_duplicates("rate_date").sort_values("rate_date")
        rates["usd_inr_return_5obs"] = rates["usd_inr_rate"].pct_change(5)
        rates["available_at"] = rates["rate_date"] + pd.Timedelta(days=1)

        bar_rows = pd.DataFrame({"bar_time": bars, "original_index": list(index)})
        aligned = pd.merge_asof(
            bar_rows.sort_values("bar_time"),
            rates[["available_at", "rate_date", "usd_inr_rate", "usd_inr_return_5obs"]].sort_values("available_at"),
            left_on="bar_time",
            right_on="available_at",
            direction="backward",
            tolerance=pd.Timedelta(days=7),
        )
        aligned = aligned.set_index("original_index")
        latest = aligned.iloc[-1]
        if pd.isna(latest["usd_inr_rate"]):
            return None, {"available": False, "reason": "No recent point-in-time USD/INR rate matches the latest market bar."}

        features = aligned[["usd_inr_rate", "usd_inr_return_5obs"]].copy()
        if pd.isna(features["usd_inr_return_5obs"].iloc[-1]):
            features = features[["usd_inr_rate"]]
        status = {
            "available": True,
            "provider": "Frankfurter reference rates",
            "reference_date": latest["rate_date"].date().isoformat(),
            "available_from": latest["available_at"].date().isoformat(),
            "usd_inr_rate": float(latest["usd_inr_rate"]),
            "usd_inr_return_5obs_pct": float(latest["usd_inr_return_5obs"] * 100) if pd.notna(latest["usd_inr_return_5obs"]) else None,
            "lag_days": 1,
            "used_as_model_feature": True,
        }
        return features, status
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        return None, {"available": False, "reason": f"Optional Frankfurter context unavailable: {exc}"}
