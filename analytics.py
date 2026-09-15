"""Data analytics and reporting for the fraud agent.

Produces the evidence package for investigators and compliance:

* PNG charts (ROC / PR curves, confusion matrix, risk distribution, fraud rate
  by category / country, transaction volume by hour, rule-flag frequency,
  disposition mix, amount distribution) written to ``output/charts``.
* Tabular summaries (priority cases, flag frequency, dispositions, monthly
  activity) returned as DataFrames and embedded in a self-contained HTML report
  together with the agent audit trail.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#b0bec5",
    "figure.dpi": 110,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
})

SEVERITY_COLORS = {"CRITICAL": "#b3261e", "HIGH": "#e37400",
                   "MEDIUM": "#bf9000", "LOW": "#1f7a3d"}


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_roc_curve(trained, path: Path) -> str:
    if not trained or not trained.tpr:
        return ""
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(trained.fpr, trained.tpr, color="#1565c0", lw=2)
    ax.plot([0, 1], [0, 1], "--", color="#90a4ae")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curve (validation, AUC={trained.metrics.get('roc_auc')})")
    _save(fig, path)
    return path.name


def chart_pr_curve(trained, path: Path) -> str:
    if not trained or not trained.precision:
        return ""
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(trained.recall, trained.precision, color="#00897b", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-recall (validation, PR-AUC={trained.metrics.get('pr_auc')})")
    ax.set_ylim(0, 1.05)
    _save(fig, path)
    return path.name


def chart_confusion_matrix(trained, path: Path) -> str:
    if not trained or "tp" not in trained.metrics:
        return ""
    m = trained.metrics
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    labels = ["Normal", "Fraud"]
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels, rotation=90, va="center")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix @ p≥{m.get('decision_threshold', 0.5):.2f}")
    _save(fig, path)
    return path.name


def chart_risk_distribution(scored: pd.DataFrame, path: Path,
                            label_col: str = "is_fraud") -> str:
    if scored.empty:
        return ""
    has_label = label_col in scored.columns and scored[label_col].nunique() > 1
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    if has_label:
        for v, color, name in ((0, "#90caf9", "Normal"),
                               (1, "#e57373", "Fraud")):
            subset = scored[scored[label_col] == v]
            if not subset.empty:
                ax.hist(subset["risk_score"], bins=30, alpha=0.6, color=color,
                        label=f"{name} (n={len(subset)})")
        ax.legend(loc="upper right")
    else:
        ax.hist(scored["risk_score"], bins=30, color="#64b5f6")
    cutoff = scored["risk_score"].mean() if scored.empty else None
    if has_label and cutoff is None:
        pass
    ax.set_xlabel("Final fraud-risk score (0-1)")
    ax.set_ylabel("Transactions")
    ax.set_title("Risk score distribution")
    _save(fig, path)
    return path.name


def chart_amount_distribution(scored: pd.DataFrame, path: Path,
                              label_col: str = "is_fraud") -> str:
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    amount = scored["amount"].to_numpy(dtype="float")
    if (amount <= 0).any():
        bins = np.histogram_bin_edges(np.log1p(np.abs(amount) + 1e-9), bins=30)
        x = np.log10(np.abs(amount) + 1e-9)
    else:
        x = np.log10(amount)
        bins = 30
    has_label = label_col in scored.columns and scored[label_col].nunique() > 1
    if has_label:
        for v, color, name in ((0, "#90caf9", "Normal"), (1, "#e57373", "Fraud")):
            subset_x = x[scored[label_col] == v]
            if len(subset_x):
                ax.hist(subset_x, bins=bins, alpha=0.6, color=color,
                        label=f"{name} (n={len(subset_x)})")
        ax.legend(loc="upper right")
    else:
        ax.hist(x, bins=bins, color="#64b5f6")
    ax.set_xlabel("log10(amount)")
    ax.set_ylabel("Transactions")
    ax.set_title("Amount distribution")
    _save(fig, path)
    return path.name


def chart_fraud_by_category(scored: pd.DataFrame, path: Path,
                            label_col: str = "is_fraud") -> str:
    has_label = label_col in scored.columns and scored[label_col].sum() > 0
    if not has_label:
        return ""
    rate = (scored.groupby("merchant_category")[label_col].agg(["sum", "count"])
            .rename(columns={"sum": "fraud", "count": "total"}))
    rate = rate[rate["total"] >= 5].copy()
    if rate.empty:
        return ""
    rate["fraud_rate"] = rate["fraud"] / rate["total"] * 100
    rate = rate.sort_values("fraud_rate").tail(12)
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ax.barh(rate.index, rate["fraud_rate"], color="#e37400")
    for y, (fr, total) in enumerate(zip(rate["fraud_rate"], rate["total"])):
        ax.text(fr + 0.3, y, f"{fr:.1f}% (n={int(total)})", va="center", fontsize=8)
    ax.set_xlabel("Fraud rate (%)")
    ax.set_title("Fraud rate by merchant category")
    _save(fig, path)
    return path.name


def chart_fraud_by_country(scored: pd.DataFrame, path: Path,
                           label_col: str = "is_fraud") -> str:
    has_label = label_col in scored.columns and scored[label_col].sum() > 0
    if not has_label:
        return ""
    counts = scored.loc[scored[label_col] == 1, "country"].value_counts().head(10)
    if counts.empty:
        return ""
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.bar(counts.index, counts.values, color="#e57373")
    ax.set_xlabel("Country")
    ax.set_ylabel("Fraud transactions")
    ax.set_title("Fraud transactions by country")
    _save(fig, path)
    return path.name


def chart_txn_by_hour(scored: pd.DataFrame, path: Path,
                      label_col: str = "is_fraud") -> str:
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    all_hours = scored.groupby("hour_of_day").size()
    ax.bar(all_hours.index, all_hours.values, color="#b0bec5", alpha=0.7,
           label="All transactions")
    if label_col in scored.columns and scored[label_col].sum() > 0:
        fraud_hours = scored[scored[label_col] == 1].groupby("hour_of_day").size()
        if not fraud_hours.empty:
            ax.bar(fraud_hours.index, fraud_hours.values, color="#c62828",
                   alpha=0.9, label="Fraud")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Transactions")
    ax.set_title("Transaction volume by hour")
    ax.legend()
    _save(fig, path)
    return path.name


def chart_flag_frequency(scored: pd.DataFrame, path: Path) -> str:
    counts = flags_table(scored)
    if counts.empty:
        return ""
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    top = counts.head(12)
    ax.barh(top["flag"], top["count"], color="#1565c0")
    ax.set_xlabel("Count")
    ax.set_title("Rule-flag frequency across the batch")
    _save(fig, path)
    return path.name


def chart_disposition_mix(scored: pd.DataFrame, path: Path) -> str:
    counts = scored["disposition"].value_counts()
    if counts.empty:
        return ""
    colors = {"APPROVE": "#1f7a3d", "REVIEW": "#bf9000", "BLOCK": "#b3261e"}
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    wedges, _, autotexts = ax.pie(
        counts.values, labels=[f"{k}\n{v}" for k, v in counts.items()],
        colors=[colors.get(k, "#90a4ae") for k in counts.index],
        autopct="%1.0f%%", startangle=90, textprops={"fontsize": 8})
    ax.set_title("Agent dispositions")
    _save(fig, path)
    return path.name


def make_charts(scored: pd.DataFrame, trained, charts_dir: Path,
                label_col: str = "is_fraud") -> List[str]:
    """Render all charts into ``charts_dir`` and return the generated filenames."""
    charts_dir.mkdir(parents=True, exist_ok=True)
    names: List[str] = []
    for fn in (
        lambda: chart_confusion_matrix(trained, charts_dir / "confusion_matrix.png"),
        lambda: chart_roc_curve(trained, charts_dir / "roc_curve.png"),
        lambda: chart_pr_curve(trained, charts_dir / "pr_curve.png"),
        lambda: chart_risk_distribution(scored, charts_dir / "risk_distribution.png", label_col),
        lambda: chart_amount_distribution(scored, charts_dir / "amount_distribution.png", label_col),
        lambda: chart_fraud_by_category(scored, charts_dir / "fraud_by_category.png", label_col),
        lambda: chart_fraud_by_country(scored, charts_dir / "fraud_by_country.png", label_col),
        lambda: chart_txn_by_hour(scored, charts_dir / "txn_by_hour.png", label_col),
        lambda: chart_flag_frequency(scored, charts_dir / "flag_frequency.png"),
        lambda: chart_disposition_mix(scored, charts_dir / "disposition_mix.png"),
    ):
        name = fn()
        if name:
            names.append(name)
    return names


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def flags_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Count rule-flag codes raised in the batch."""
    rows = []
    for codes in scored["flags"].dropna():
        for code in str(codes).split(","):
            code = code.strip()
            if code and code != "-":
                rows.append(code)
    if not rows:
        return pd.DataFrame(columns=["flag", "count"])
    return (pd.Series(rows).value_counts()
            .rename_axis("flag").reset_index(name="count"))


def disposition_table(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.groupby("disposition").agg(
        count=("txn_id", "size"),
        value=("amount", "sum"),
        avg_risk=("risk_score", "mean"),
    ).reset_index()
    return out


def cases_table(scored: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    col = "risk_score"
    ranked = scored.sort_values(col, ascending=False)
    table = ranked.head(limit)[[
        "txn_datetime", "txn_id", "account_id", "merchant", "merchant_category",
        "country", "channel", "amount", "risk_score", "model_proba", "anomaly_score",
        "rule_score", "disposition", "severity", "flags",
    ]].copy()
    table = table.rename(columns={
        "txn_datetime": "date", "merchant_category": "category",
        "risk_score": "risk", "model_proba": "model_prob",
        "anomaly_score": "anomaly", "rule_score": "rules",
    })
    return table


def monthly_table(scored: pd.DataFrame) -> pd.DataFrame:
    df = scored.copy()
    df["month"] = pd.to_datetime(df["txn_datetime"]).dt.to_period("M").astype(str)
    out = df.groupby("month").agg(
        transactions=("txn_id", "size"),
        value=("amount", "sum"),
        flagged=("disposition", lambda s: int((s != "APPROVE").sum())),
        blocked=("disposition", lambda s: int((s == "BLOCK").sum())),
        avg_risk=("risk_score", "mean"),
    ).reset_index()
    return out


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------
def _img(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _df_to_html(df: pd.DataFrame, klass: str = "tbl") -> str:
    if df.empty:
        return "<p class='muted'>No rows.</p>"
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].round(3)
    return df.to_html(index=False, classes=klass, escape=True, border=0)


def render_html_report(*, summary: Dict, scored: pd.DataFrame,
                       trained, charts_dir: Path, trail: List[Dict],
                       tables: Dict[str, pd.DataFrame], title: str) -> str:
    charts = sorted(p.name for p in charts_dir.glob("*.png")) if charts_dir.exists() else []

    chart_html = "".join(
        f"<figure><img src='{_img(charts_dir / name)}' alt='{name}'>"
        f"<figcaption>{name.replace('_', ' ')}</figcaption></figure>"
        for name in charts
    )

    trail_html = "".join(
        f"<tr><td>{s.get('step', '')}</td><td>{s.get('action', '')}</td>"
        f"<td>{s.get('detail', '')}</td></tr>"
        for s in trail
    )

    table_html = "".join(
        f"<h3>{label}</h3>{_df_to_html(t)}"
        for label, t in tables.items()
    )

    metrics = ""
    if trained is not None:
        m = trained.metrics
        metrics = "<h3>Model performance (chronological validation)</h3><table class='tbl'>" + "".join(
            f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in m.items()
        ) + "</table>"

    summary_rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>"
                           for k, v in summary.items())

    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{title}</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#263238;background:#fafafa}}
h1,h2,h3{{color:#0d47a1}} h2{{border-bottom:2px solid #e0e0e0;padding-bottom:4px;margin-top:36px}}
.cards{{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0}}
.card{{background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:12px 18px;min-width:150px}}
.card b{{display:block;font-size:20px;color:#0d47a1}} .card span{{font-size:11px;color:#78909c}}
figure{{margin:8px;display:inline-block;text-align:center}}
figure img{{max-width:420px;border:1px solid #e0e0e0;border-radius:6px}}
figcaption{{font-size:11px;color:#78909c}}
.tbl{{border-collapse:collapse;width:100%;margin:10px 0;background:#fff}}
.tbl th,.tbl td{{border:1px solid #e0e0e0;padding:6px 9px;text-align:left;font-size:12px}}
.tbl th{{background:#e3f2fd;font-weight:600}}
.muted{{color:#78909c}} pre{{background:#f5f5f5;padding:10px;font-size:11px}}
</style></head><body>
<h1>{title}</h1>
<p class='muted'>Generated by the Fraud Detection Agent &middot; {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
<h2>Executive summary</h2>
<table class='tbl'>" + summary_rows + "</table>
<div class='cards'>
<div class='card'><b>{summary.get('screened', 0)}</b><span>transactions screened</span></div>
<div class='card'><b>{summary.get('flagged', 0)}</b><span>flagged for review or block</span></div>
<div class='card'><b>{summary.get('blocked', 0)}</b><span>blocked</span></div>
<div class='card'><b>{summary.get('fraud_expected', 'n/a')}</b><span>fraud expected in batch</span></div>
</div>
<h2>Charts</h2>""" + chart_html + """
<h2>Tables</h2>""" + table_html + """
<h2>Model performance</h2>""" + metrics + """
<h2>Agent audit trail</h2>
<table class='tbl'><tr><th>Step</th><th>Action</th><th>Detail</th></tr>""" + trail_html + """
<h2>Methodology</h2>
<p>Each transaction is screened in five layers:</p>
<pre>
1. LOAD     - parse and normalise the input file.
2. FEATURE  - build behaviour features (velocity windows, amount deviation
              from the account baseline, merchant / country / channel risk,
              time-of-day patterns, new-merchant flags).
3. LEARN    - fit an Isolation Forest anomaly detector and (when labelled
              history exists) a Random Forest classifier on historical cases.
4. SCORE    - combine 65% supervised probability + 20% anomaly score +
              15% rule-engine score into a single 0-1 fraud-risk score.
5. DECIDE   - BLOCK when risk is high, REVIEW when the model or rules
              are uncertain, otherwise APPROVE; every decision is recorded
              with its evidence in the audit trail.
</pre>
<p class='muted'>Alerts and scores are decision support only. Investigate each
flagged case against supporting documents before acting.</p>
</body></html>"""