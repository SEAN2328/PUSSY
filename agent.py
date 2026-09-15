"""The Fraud Detection Agent - an autonomous screening pipeline.

``FraudAgent.run()`` executes the full sequence independently:

    LOAD -> PROFILE -> FEATURE -> LEARN -> SCORE -> DECIDE
    -> EVALUATE -> AUDIT -> ANALYSE -> OUTPUT

It reads a labelled *history* file (cases already investigated by the finance
team) and an unlabelled *live* batch, learns from the history, then screens the
live batch. Every score, flag and disposition is written to an audit trail, and
the analytics stage produces PNG charts, CSV/XLSX-style tables and a
self-contained HTML report.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import analytics as ana
from . import models as mdl
from .engine import FraudConfig, parse_transactions_csv, screen
from .features import FEATURE_COLUMNS, build_features


class FraudAgentError(RuntimeError):
    """Raised for invalid inputs or a failed pipeline stage."""


@dataclass
class AgentConfig:
    """Tunable policy for the screening agent."""
    review_threshold: float = 0.55
    block_threshold: float = 0.85
    target_recall: float = 0.80
    anomaly_contamination: float = 0.03
    history_label_col: str = "is_fraud"
    true_label_col: str = "true_fraud"
    seed: int = 42
    engine_config: FraudConfig = field(default_factory=FraudConfig)


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        if v is None or pd.isna(v):
            return float("nan")
        return float("nan")


class FraudAgent:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.trail: List[Dict] = []

    def _log(self, step: int, action: str, detail: str) -> None:
        self.trail.append({"step": step, "action": action, "detail": detail})

    # ------------------------------------------------------------------
    # Stage 1-2: LOAD + PROFILE
    # ------------------------------------------------------------------
    def _load(self, path: Optional[Path]) -> pd.DataFrame:
        if path is None or not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        df = _normalise_schema(df)
        if df.empty:
            raise FraudAgentError(f"no transaction rows could be parsed from {path}")
        return df

    def _profile(self, df: pd.DataFrame, name: str) -> dict:
        dates = pd.to_datetime(df["txn_datetime"], errors="coerce")
        return {
            "period": f"{dates.min().date()} to {dates.max().date()}",
            "transactions": int(len(df)),
            "value": float(df["amount"].sum()),
            "accounts": int(df["account_id"].nunique()),
        }

    # ------------------------------------------------------------------
    # Stage 3: FEATURE
    # ------------------------------------------------------------------
    def _regression_step(self, hist: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
        combined = pd.concat([hist, live], ignore_index=True)
        combined["part"] = ["history"] * len(hist) + ["live"] * len(live)
        combined["_row"] = np.arange(len(combined))
        return combined

    # ------------------------------------------------------------------
    # Stage 4: LEARN (supervised + anomaly)
    # ------------------------------------------------------------------
    def _learn(self, combined: pd.DataFrame,
               trained: Optional[mdl.TrainOutput],
               anom: mdl.AnomalyOutput) -> pd.DataFrame:
        combined["anomaly_score"] = anom.normalized_score
        if trained is not None:
            combined["model_proba"] = mdl.predict_proba(
                trained.model, combined, trained.feature_cols)
        else:
            combined["model_proba"] = np.nan
        return combined

    # ------------------------------------------------------------------
    # Stage 5-6: SCORE + DECIDE
    # ------------------------------------------------------------------
    def _run_rules(self, live: pd.DataFrame) -> pd.DataFrame:
        if live.empty:
            return live
        csv_text = _frame_to_engine_csv(live)
        result = screen(parse_transactions_csv(csv_text), self.config.engine_config)
        by_ref = {t.reference: t for t in result.transactions}
        rules = np.zeros(len(live), dtype="float")
        flags: List[str] = []
        reasons: List[str] = []
        for i, txn_id in enumerate(live["txn_id"]):
            t = by_ref.get(str(txn_id))
            if t is None:
                rules[i] = 0.0
                flags.append("-")
                reasons.append("no rule signals")
                continue
            rules[i] = float(t.score)
            flags.append(", ".join(f.code for f in t.flags) or "-")
            reasons.append("; ".join(f.detail for f in t.flags) or "No rule signals")
        live = live.copy()
        live["rule_score"] = rules
        live["flags"] = flags
        live["rule_reason"] = reasons
        n_flags = int(sum(1 for f in flags if f != "-"))
        self._log(6, "Analyse - rules",
                  f"{n_flags} rule evidence flag(s) raised across the live batch")
        return live

    def _score_and_decide(self, live: pd.DataFrame,
                          trained: Optional[mdl.TrainOutput]) -> pd.DataFrame:
        combined = live  # features and model scores already attached
        supervised_arr = combined["model_proba"].to_numpy(dtype="float")
        supervised = None if np.isnan(supervised_arr).any() else supervised_arr
        if supervised is not None:
            combined["model_proba"] = supervised
        rule = combined["rule_score"].to_numpy(dtype="float")
        anom = combined["anomaly_score"].to_numpy(dtype="float")
        risk = mdl.blend_risk(supervised, anom, rule)
        combined["risk_score"] = risk

        sev = []
        disp = []
        for r in risk:
            if r >= 0.8:
                sev.append("CRITICAL")
            elif r >= 0.55:
                sev.append("HIGH")
            elif r >= 0.30:
                sev.append("MEDIUM")
            else:
                sev.append("LOW")
            if r >= self.config.block_threshold:
                disp.append("BLOCK")
            elif r >= self.config.review_threshold:
                disp.append("REVIEW")
            else:
                disp.append("APPROVE")
        combined["severity"] = sev
        combined["disposition"] = disp
        combined["reason"] = _build_reason(combined, trained is not None)
        return combined

    # ------------------------------------------------------------------
    # Stage 7: EVALUATE
    # ------------------------------------------------------------------
    def _evaluate(self, scored: pd.DataFrame) -> dict:
        true_label = self.config.true_label_col
        if true_label not in scored.columns:
            return {}
        y = scored[true_label].astype(int).to_numpy()
        flagged = (scored["disposition"] != "APPROVE").to_numpy()
        caught = int(((y == 1) & flagged).sum())
        total_fraud = int(y.sum())
        flagged_rows = int(flagged.sum())
        precision = caught / flagged_rows if flagged_rows else 0.0
        recall = caught / total_fraud if total_fraud else 0.0
        value_at_risk = float(scored.loc[flagged, "amount"].sum())
        return {
            "true_fraud_in_batch": total_fraud,
            "fraud_caught": caught,
            "fraud_missed": int(total_fraud - caught),
            "recall": round(recall, 4) if total_fraud else 0.0,
            "precision": round(precision, 4) if total_fraud else 0.0,
            "review_block_value_at_risk": round(value_at_risk, 2),
        }

    # ------------------------------------------------------------------
    # Stage 8-10: AUDIT + ANALYSE + OUTPUT
    # ------------------------------------------------------------------
    def _write_outputs(self, scored: pd.DataFrame, trained, profile: dict,
                       evaluate: dict, output_dir: Path) -> dict:
        out = output_dir
        charts = out / "charts"
        charts.mkdir(parents=True, exist_ok=True)

        summary = _build_summary(scored, profile, evaluate)

        trail = list(self.trail)

        # Tables
        tables = {
            "Disposition summary": ana.disposition_table(scored),
            "Priority cases": ana.cases_table(scored, limit=25),
            "Rule-flag frequency": ana.flags_table(scored),
            "Monthly activity": ana.monthly_table(scored) if len(scored) else
                                pd.DataFrame(),
        }

        chart_names = ana.make_charts(scored, trained, charts,
                                      label_col=self.config.true_label_col)

        html = ana.render_html_report(
            summary=summary, scored=scored, trained=trained,
            charts_dir=charts, trail=trail, tables=tables,
            title="Fraud Detection Agent - Screening Report")

        # Persist artifacts
        (out / "audit_trail.json").write_text(
            json.dumps({"trail": trail, "summary": summary,
                        "evaluation": evaluate}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        (out / "summary_stats.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "report.html").write_text(html, encoding="utf-8")

        scored.to_csv(out / "live_scored.csv", index=False)
        ana.cases_table(scored, limit=100).to_csv(out / "flagged_cases.csv", index=False)
        ana.disposition_table(scored).to_csv(out / "disposition_summary.csv", index=False)
        ana.flags_table(scored).to_csv(out / "flag_frequency.csv", index=False)
        if trained is not None:
            (out / "model_metrics.json").write_text(
                json.dumps(trained.json_compatible(), indent=2, ensure_ascii=False),
                encoding="utf-8")

        summary["charts"] = chart_names
        summary["csv_files"] = sorted(p.name for p in out.glob("*.csv"))
        return summary

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def run(self, history_path: Path, live_path: Path,
            output_dir: Path) -> Dict[str, object]:
        """Run the complete agent pipeline and return the summary + artifacts."""
        self.trail = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. LOAD
        hist = self._load(Path(history_path))
        live = self._load(Path(live_path))
        if live.empty:
            raise FraudAgentError("no live transactions to screen")
        if self.config.history_label_col in live.columns:
            live = live.rename(columns={
                self.config.history_label_col: self.config.true_label_col})
        self._log(1, "Load",
                  f"loaded {len(hist)} history row(s) and {len(live)} live row(s)")

        # 2. PROFILE
        hist_profile = self._profile(hist, "history")
        live_profile = self._profile(live, "live")
        self._log(2, "Profile",
                  f"history {hist_profile['period']} ({hist_profile['transactions']} txns); "
                  f"live {live_profile['period']} ({live_profile['transactions']} txns)")

        # 3. FEATURE (combined so live rows get historical context)
        combined = self._regression_step(hist, live)
        combined = build_features(combined)
        self._log(3, "Feature",
                  f"engineered {len(FEATURE_COLUMNS)} behaviour features per "
                  f"transaction from account baselines, trailing velocity "
                  f"windows, merchant/country/channel risk and time-of-day "
                  f"patterns")

        # 4. LEARN
        trained = None
        label = self.config.history_label_col
        if label in combined.columns and combined[label].notna().any():
            trained = mdl.train_classifier(
                combined, FEATURE_COLUMNS, label_col=label,
                target_recall=self.config.target_recall, seed=self.config.seed)
            self._log(4, "Learn", f"trained Random Forest on {trained.metrics['n_train']} "
                                  f"labelled cases; validation ROC-AUC "
                                  f"{trained.metrics['roc_auc']}, PR-AUC "
                                  f"{trained.metrics['pr_auc']}")
        else:
            self._log(4, "Learn",
                      "no labelled history supplied - using anomaly detection + rules only")

        anom = mdl.fit_anomaly(combined, FEATURE_COLUMNS,
                               contamination=self.config.anomaly_contamination,
                               seed=self.config.seed)
        self._log(5, "Anomaly",
                  f"Isolation Forest {len(combined)} rows, contamination "
                  f"{self.config.anomaly_contamination}")

        combined = self._learn(combined, trained, anom)

        # 5. SCORE (rules + blend)
        live_idx = combined.index[combined["part"] == "live"]
        live = combined.loc[live_idx].copy().reset_index(drop=True)
        live = self._run_rules(live)
        live = self._score_and_decide(live, trained)
        self._log(7, "Score",
                  f"blended supervised probability + anomaly + rules into a 0-1 "
                  f"risk score for {len(live)} live transaction(s)")

        # 6. DECIDE
        self._log(8, "Decide",
                  f"{int((live['disposition'] == 'BLOCK').sum())} blocked, "
                  f"{int((live['disposition'] == 'REVIEW').sum())} queued for review, "
                  f"{int((live['disposition'] == 'APPROVE').sum())} approved")

        # 7. EVALUATE (hidden-labels demo)
        evaluate = self._evaluate(live)
        if evaluate:
            self._log(9, "Evaluate",
                      f"caught {evaluate['fraud_caught']}/{evaluate['true_fraud_in_batch']} "
                      f"true fraud (recall {evaluate['recall']}, precision "
                      f"{evaluate['precision']})")

        # 8-10. AUDIT + ANALYSE + OUTPUT
        self._log(10, "Audit",
                  f"wrote audit trail with {len(self.trail)} steps to "
                  f"{output_dir / 'audit_trail.json'}")
        summary = self._write_outputs(live, trained, live_profile, evaluate, output_dir)
        self._log(11, "Complete",
                  f"screening package written to {output_dir}")

        return {"summary": summary, "trail": self.trail, "evaluation": evaluate,
                "trained": trained, "scored": live, "output_dir": output_dir}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalise_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Map flexible column headers onto the canonical schema."""
    rename = {
        "transaction_id": "txn_id", "id": "txn_id",
        "transaction_datetime": "txn_datetime", "datetime": "txn_datetime",
        "date": "txn_datetime", "acct_id": "account_id", "customer_id": "account_id",
        "merchant_name": "merchant", "category": "merchant_category",
        "type": "merchant_category", "card_channel": "channel",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "txn_datetime" not in df.columns:
        raise FraudAgentError("missing a datetime column (txn_datetime / date)")
    df["txn_datetime"] = pd.to_datetime(df["txn_datetime"], errors="coerce")
    df = df.dropna(subset=["txn_datetime"])
    if "amount" not in df.columns:
        raise FraudAgentError("missing an amount column")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])
    for col in ("merchant", "merchant_category", "country", "channel"):
        if col not in df.columns:
            df[col] = "unknown"
    for col in ("account_id", "txn_id"):
        df[col] = df[col].fillna("unknown") if col in df.columns else "unknown"
    return df


def _frame_to_engine_csv(df: pd.DataFrame) -> str:
    lines = ["date,reference,description,payee,category,amount"]
    for r in df.itertuples(index=False):
        when = pd.to_datetime(r.txn_datetime)
        lines.append(f"{when.strftime('%Y-%m-%d')},{r.txn_id},{r.merchant},"
                     f"{r.merchant},{r.merchant_category},{r.amount}")
    return "\n".join(lines)


def _build_reason(df: pd.DataFrame, supervised: bool) -> List[str]:
    reasons = []
    for r in df.itertuples(index=False):
        bits = []
        proba = getattr(r, "model_proba", np.nan)
        if supervised and not np.isnan(_safe_float(proba)):
            bits.append(f"model probability {float(proba):.2f}")
        anomaly = getattr(r, "anomaly_score", np.nan)
        if not np.isnan(_safe_float(anomaly)):
            bits.append(f"anomaly score {float(anomaly):.2f}")
        rules = getattr(r, "rule_reason", "")
        if rules and rules != "No rule signals":
            bits.append(f"rules: {rules}")
        if not bits:
            bits.append("all signals within normal range")
        reasons.append("; ".join(bits))
    return reasons


def _build_summary(scored: pd.DataFrame, profile: dict, evaluate: dict) -> dict:
    flagged = scored[scored["disposition"] != "APPROVE"]
    summary = {
        "period": profile["period"],
        "screened": int(profile["transactions"]),
        "value": round(profile["value"], 2),
        "accounts": int(profile["accounts"]),
        "approved": int((scored["disposition"] == "APPROVE").sum()),
        "review": int((scored["disposition"] == "REVIEW").sum()),
        "blocked": int((scored["disposition"] == "BLOCK").sum()),
        "flagged": int(len(flagged)),
        "flagged_value": round(float(flagged["amount"].sum()), 2),
        "flag_rate_pct": round(len(flagged) / len(scored) * 100, 2) if len(scored) else 0.0,
        "avg_risk": round(float(scored["risk_score"].mean()), 4),
        "max_risk": round(float(scored["risk_score"].max()), 4),
    }
    summary.update(evaluate)
    return summary