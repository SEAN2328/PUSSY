"""Feature engineering for the fraud model.

Builds a leakage-safe numeric feature table from raw transaction rows. Only
information available *before* a transaction is used for account-behaviour
features (expanding means, trailing time windows), so the resulting features are
safe for both training and production screening.

Expected raw columns: ``account_id``, ``txn_datetime``, ``amount``, ``merchant``,
``merchant_category``, ``country``, ``channel`` (``is_fraud`` optional).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .risk import CATEGORY_RISK, CHANNEL_RISK, COUNTRY_RISK

# Order-stable list of features consumed by the models and the reports.
FEATURE_COLUMNS = [
    "amount_log",
    "hour_of_day",
    "is_weekend",
    "is_night",
    "category_risk",
    "country_risk",
    "channel_risk",
    "acct_amt_mean_prior",
    "acct_amt_std_prior",
    "acct_amount_zscore",
    "amount_vs_acct_mean",
    "txn_count_1d_acct",
    "txn_amount_1d_acct",
    "txn_count_7d_acct",
    "txn_amount_7d_acct",
    "n_merchants_7d_acct",
    "txn_count_same_merchant_24h",
    "is_new_merchant_acct",
    "is_new_country_acct",
    "hours_since_prev_acct",
]

_TIME_COLS = ["txn_count_1d_acct", "txn_amount_1d_acct", "txn_count_7d_acct",
              "txn_amount_7d_acct", "n_merchants_7d_acct",
              "txn_count_same_merchant_24h"]


def _prior_rolling(values: pd.Series, window: str, agg: str) -> pd.Series:
    """Trailing-window statistic that excludes the current row.

    ``values`` must be a numeric series indexed by a sorted DatetimeIndex;
    only its values matter, ordering matches the group.
    """
    rolled = values.rolling(window, closed="both")
    if agg == "count":
        result = rolled.count() - 1
    elif agg == "sum":
        result = rolled.sum() - values
    elif agg == "merchants":
        result = rolled.apply(lambda x: int(np.unique(x).size), raw=True) - 1
    else:
        raise ValueError(f"unknown agg: {agg}")
    return pd.Series(result.values, index=values.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered risk features to a transaction frame (in place-safe).

    The frame must already contain ``txn_datetime`` (datetime) and the raw
    transaction columns. Returns the same frame augmented with the feature
    columns; any rows with an unparseable amount are dropped.
    """
    out = df.copy()
    out["txn_datetime"] = pd.to_datetime(out["txn_datetime"])
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out = out.dropna(subset=["amount", "account_id", "txn_datetime"])
    out = out.sort_values(["account_id", "txn_datetime"]).reset_index(drop=True)

    out["amount_log"] = np.log1p(out["amount"].clip(lower=0.0))
    out["hour_of_day"] = out["txn_datetime"].dt.hour
    out["is_weekend"] = (out["txn_datetime"].dt.dayofweek >= 5).astype(int)
    out["is_night"] = ((out["hour_of_day"] < 7) | (out["hour_of_day"] > 22)).astype(int)
    out["category_risk"] = out["merchant_category"].map(CATEGORY_RISK).fillna(0.3)
    out["country_risk"] = out["country"].map(COUNTRY_RISK).fillna(0.5)
    out["channel_risk"] = out["channel"].map(CHANNEL_RISK).fillna(0.5)

    grp = out.groupby("account_id", sort=False)
    out["acct_amt_mean_prior"] = grp["amount"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean())
    out["acct_amt_std_prior"] = grp["amount"].transform(
        lambda s: s.shift(1).expanding(min_periods=2).std())
    out["acct_amt_std_prior"] = out["acct_amt_std_prior"].fillna(0.0)
    out["acct_amount_zscore"] = (
        (out["amount"] - out["acct_amt_mean_prior"])
        / out["acct_amt_std_prior"].replace(0.0, np.nan)
    ).fillna(0.0)
    out["amount_vs_acct_mean"] = (
        out["amount"] / out["acct_amt_mean_prior"].replace(0.0, np.nan)
    ).fillna(1.0)
    out["acct_amt_mean_prior"] = out["acct_amt_mean_prior"].fillna(
        out["amount"])

    out["is_new_merchant_acct"] = (grp["merchant"].cumcount() == 0).astype(int)
    out["is_new_country_acct"] = (grp["country"].cumcount() == 0).astype(int)
    out["hours_since_prev_acct"] = (
        grp["txn_datetime"].diff().dt.total_seconds() / 3600.0
    )

    piece = []
    for _, sub in out.groupby("account_id", sort=False):
        s = sub.set_index("txn_datetime").sort_index()
        codes = pd.Series(
            pd.factorize(s["merchant"])[0], index=s.index, dtype="int64")
        sub["txn_count_1d_acct"] = _prior_rolling(s["amount"], "1D", "count").values
        sub["txn_amount_1d_acct"] = _prior_rolling(s["amount"], "1D", "sum").values
        sub["txn_count_7d_acct"] = _prior_rolling(s["amount"], "7D", "count").values
        sub["txn_amount_7d_acct"] = _prior_rolling(s["amount"], "7D", "sum").values
        sub["n_merchants_7d_acct"] = _prior_rolling(codes, "7D", "merchants").values
        piece.append(sub)
    out = pd.concat(piece).sort_values(["account_id", "txn_datetime"])

    merchant_piece = []
    for _, sub in out.groupby(["account_id", "merchant"], sort=False):
        s = sub.set_index("txn_datetime").sort_index()
        sub["txn_count_same_merchant_24h"] = _prior_rolling(
            s["amount"], "24h", "count").values
        merchant_piece.append(sub)
    out = pd.concat(merchant_piece).sort_values(["account_id", "txn_datetime"])

    for col in _TIME_COLS:
        out[col] = out[col].fillna(0)
    out["hours_since_prev_acct"] = out["hours_since_prev_acct"].fillna(
        24 * 30).clip(lower=0, upper=24 * 60)
    out["amount_vs_acct_mean"] = out["amount_vs_acct_mean"].clip(lower=0, upper=50)

    return out.reset_index(drop=True)