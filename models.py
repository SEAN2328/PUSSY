"""Machine-learning models for the fraud agent.

Two complementary models:

* **Supervised** - a Random Forest trained on historically reviewed (labelled)
  fraud cases; outputs a fraud probability and validation metrics (ROC AUC,
  PR AUC, confusion matrix at the tuned threshold).
* **Unsupervised** - an Isolation Forest that scores how anomalous each
  transaction is even when no labels are available.

All models are deterministic (fixed seeds). Threshold selection targets a
configurable recall so the agent favours catching fraud over restricting alert
volume, while reporting precision to the investigator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainOutput:
    """Everything the supervised stage produces."""
    model: object
    feature_cols: List[str]
    metrics: Dict = field(default_factory=dict)
    thresholds: List[float] = field(default_factory=list)
    tpr: List[float] = field(default_factory=list)
    fpr: List[float] = field(default_factory=list)
    precision: List[float] = field(default_factory=list)
    recall: List[float] = field(default_factory=list)
    decision_threshold: float = 0.5

    def json_compatible(self) -> dict:
        out = dict(self.metrics)
        out["decision_threshold"] = round(self.decision_threshold, 4)
        return out


def split_by_time(df: pd.DataFrame, frac: float = 0.8, seed: int = 42):
    """Chronological train / validation split (no future leakage)."""
    sorted_df = df.sort_values("txn_datetime")
    cut = int(len(sorted_df) * frac)
    return sorted_df.iloc[:cut].copy(), sorted_df.iloc[cut:].copy()


def _scalar(value) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def train_classifier(features: pd.DataFrame, feature_cols: List[str],
                     label_col: str = "is_fraud", target_recall: float = 0.80,
                     seed: int = 42) -> TrainOutput:
    """Train + validate the supervised model on labelled history."""
    labeled = features.dropna(subset=[label_col]).copy()
    labeled["is_fraud"] = labeled[label_col].astype(int)
    train, val = split_by_time(labeled, frac=0.8, seed=seed)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=15, class_weight="balanced",
        n_jobs=-1, random_state=seed,
    )
    clf.fit(train[feature_cols], train["is_fraud"])
    proba_val = clf.predict_proba(val[feature_cols])[:, 1]
    y_val = val["is_fraud"].to_numpy()

    fpr, tpr, thr = roc_curve(y_val, proba_val)
    precision, recall, _ = precision_recall_curve(y_val, proba_val)

    # Choose the operating threshold: best precision at target recall.
    decision = 0.5
    best_prec = -1.0
    for p, r, t in zip(precision, recall, thr):
        if r >= target_recall and p > best_prec:
            best_prec = p
            decision = float(t)

    tn, fp, fn, tp = confusion_matrix(y_val, (proba_val >= decision).astype(int)).ravel()
    metrics = {
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "val_positives": int(y_val.sum()),
        "roc_auc": round(float(roc_auc_score(y_val, proba_val)), 4),
        "pr_auc": round(float(average_precision_score(y_val, proba_val)), 4),
        "target_recall": target_recall,
        "decision_recall": round(float(tp / (tp + fn)), 4) if (tp + fn) else 0.0,
        "decision_precision": round(float(tp / (tp + fp)), 4) if (tp + fp) else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return TrainOutput(
        model=clf,
        feature_cols=feature_cols,
        metrics=metrics,
        thresholds=[float(x) for x in thr],
        tpr=[float(x) for x in tpr],
        fpr=[float(x) for x in fpr],
        precision=[float(x) for x in precision],
        recall=[float(x) for x in recall],
        decision_threshold=decision,
    )


@dataclass
class AnomalyOutput:
    scaler: object
    model: object
    normalized_score: np.ndarray  # 0..1, higher = more anomalous
    contamination: float


def fit_anomaly(features: pd.DataFrame, feature_cols: List[str],
                contamination: float = 0.03, seed: int = 42) -> AnomalyOutput:
    """Fit an Isolation Forest and return per-row anomaly scores (0..1)."""
    X = features[feature_cols].to_numpy(dtype="float")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    iso = IsolationForest(contamination=contamination, random_state=seed, n_jobs=-1)
    iso.fit(Xs)
    raw = iso.decision_function(Xs)
    lo, hi = float(raw.min()), float(raw.max())
    normalized = (1.0 - (raw - lo) / (hi - lo)).clip(0.0, 1.0)
    return AnomalyOutput(scaler=scaler, model=iso,
                         normalized_score=normalized, contamination=contamination)


def predict_proba(model: object, features: pd.DataFrame,
                  feature_cols: List[str]) -> np.ndarray:
    return model.predict_proba(features[feature_cols])[:, 1]


def anomaly_scores(anom: AnomalyOutput, features: pd.DataFrame,
                   feature_cols: List[str]) -> np.ndarray:
    Xs = anom.scaler.transform(features[feature_cols].to_numpy(dtype="float"))
    raw = anom.model.decision_function(Xs)
    return np.asarray(raw, dtype="float")

def normalize(raw: np.ndarray) -> np.ndarray:
    lo, hi = float(raw.min()), float(raw.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(raw)
    return (1.0 - (raw - lo) / (hi - lo)).clip(0.0, 1.0)


@dataclass
class BlendWeights:
    supervised: float = 0.65
    anomaly: float = 0.20
    rules: float = 0.15


def blend_risk(supervised_proba: Optional[np.ndarray],
               anomaly_norm: np.ndarray,
               rule_score: np.ndarray,
               weights: Optional[BlendWeights] = None) -> np.ndarray:
    """Final 0..1 fraud-risk score combining all three signals."""
    w = weights or BlendWeights()
    total = 0.0
    score = np.zeros(len(anomaly_norm), dtype="float")
    if supervised_proba is not None:
        score = score + w.supervised * np.asarray(supervised_proba, dtype="float")
        total += w.supervised
    score = score + w.anomaly * np.asarray(anomaly_norm, dtype="float")
    total += w.anomaly
    score = score + w.rules * (np.asarray(rule_score, dtype="float") / 100.0)
    total += w.rules
    return np.clip(score / total, 0.0, 1.0)