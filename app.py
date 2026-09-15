"""Fraud Detection Agent - Streamlit dashboard.

Run locally:   streamlit run app.py  (use the project .venv)

The dashboard drives the full agent pipeline (LOAD -> PROFILE -> FEATURE ->
LEARN -> SCORE -> DECIDE -> EVALUATE -> AUDIT -> ANALYSE -> OUTPUT) over the
bundled synthetic dataset or uploaded history/live CSVs, then presents the
model performance, dispositions, charts, priority cases, audit trail and
downloadable evidence package. All processing happens locally.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from frauddetect import AgentConfig, FraudAgent  # noqa: E402

st.set_page_config(page_title="Fraud Detection Agent", page_icon=":mag:",
                   layout="wide", initial_sidebar_state="expanded")

CHART_LABELS = {
    "confusion_matrix.png": "Confusion matrix (validation)",
    "roc_curve.png": "ROC curve (validation)",
    "pr_curve.png": "Precision-recall curve (validation)",
    "risk_distribution.png": "Risk score distribution",
    "amount_distribution.png": "Amount distribution",
    "fraud_by_category.png": "Fraud rate by merchant category",
    "fraud_by_country.png": "Fraud by country",
    "txn_by_hour.png": "Transaction volume by hour",
    "flag_frequency.png": "Rule-flag frequency",
    "disposition_mix.png": "Agent dispositions",
}


def run_agent(history, live, output, config) -> dict:
    agent = FraudAgent(config)
    result = agent.run(history, live, output)
    return result


def main() -> None:
    st.title(":mag: Fraud Detection Agent")
    st.caption(
        "Autonomous payment-fraud screening: learns from historically labelled "
        "fraud cases, then loads, profiles, scores, flags and blocks transactions "
        "in a fresh batch - with a full audit trail and data analytics (graphs + "
        "tables)."
    )

    with st.sidebar:
        st.header("Data source")
        mode = st.radio("Input mode", ["Bundled synthetic demo (history + live)",
                                       "My own history + live CSVs"])
        if mode.startswith("My own"):
            history_file = st.file_uploader("Labelled history CSV",
                                            type=["csv"], key="hist")
            live_file = st.file_uploader("Live batch CSV (no labels needed)",
                                         type=["csv"], key="live")
            out_dir = st.text_input("Output directory",
                                    str(HERE / "output"))
            has_data = history_file is not None and live_file is not None
        else:
            history_file = live_file = None
            out_dir = st.text_input("Output directory", str(HERE / "output"))
            has_data = True

        st.divider()
        st.caption("Policy")
        with st.expander("Agent thresholds", expanded=True):
            review = st.slider("Review threshold", 0.0, 0.99, 0.55, 0.01)
            block = st.slider("Block threshold", 0.0, 0.99, 0.85, 0.01)
            recall_t = st.slider("Target recall (training)", 0.5, 1.0, 0.8, 0.05)

        run_clicked = st.button("Run fraud screening agent", type="primary")

    if not has_data:
        st.info("Upload a labelled history CSV and a live batch CSV to begin.")
        st.stop()

    if not run_clicked:
        st.success("Ready. Press **Run fraud screening agent** to start the pipeline.")
        st.stop()

    output = Path(out_dir)
    config = AgentConfig(review_threshold=review, block_threshold=block,
                         target_recall=recall_t)

    if history_file is not None:
        hist_path = output / "uploads" / history_file.name
        live_path = output / "uploads" / live_file.name
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        hist_path.write_bytes(history_file.getvalue())
        live_path.write_bytes(live_file.getvalue())
    else:
        hist_path = HERE / "data" / "history.csv"
        live_path = HERE / "data" / "live.csv"

    with st.spinner("Running the agent pipeline..."):
        try:
            result = run_agent(hist_path, live_path, output, config)
        except Exception as exc:  # pragma: no cover
            st.error(f"The agent could not complete a stage: {exc}")
            st.stop()

    s = result["summary"]
    ev = result["evaluation"]
    trail = result["trail"]

    st.subheader("Screening overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions screened", f"{s['screened']:,}",
              f"{s['accounts']:,} accounts")
    c2.metric("Total value", f"${s['value']:,.2f}")
    c3.metric("Flagged for action", s["flagged"], f"{s['flag_rate_pct']}% of batch")
    c4.metric("Blocked / Review", f"{s['blocked']} / {s['review']}")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Approved", s["approved"])
    c6.metric("Avg risk", f"{s['avg_risk']:.3f}")
    if ev:
        c7.metric("Fraud caught", f"{ev['fraud_caught']}/{ev['true_fraud_in_batch']}",
                  f"recall {ev['recall']:.0%}")
        c8.metric("Precision on batch", f"{ev['precision']:.0%}")

    st.markdown(f"**Value at risk:** ${s['flagged_value']:,.2f} across "
                f"{s['flagged']} flagged transaction(s).")

    tabs = st.tabs(["Charts", "Model performance", "Priority cases",
                    "Score table", "Agent trail", "Downloads"])

    charts_dir = output / "charts"
    with tabs[0]:
        st.subheader("Data analytics (graphs)")
        if charts_dir.exists():
            names = [p.name for p in sorted(charts_dir.glob("*.png"))]
            for i in range(0, len(names), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(names):
                        name = names[i + j]
                        with cols[j]:
                            st.image(str(charts_dir / name),
                                     caption=CHART_LABELS.get(name, name))
        else:
            st.info("No charts were generated for this run.")

    with tabs[1]:
        st.subheader("Model performance (chronological validation)")
        metrics_path = output / "model_metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows = [{"metric": k, "value": v} for k, v in metrics.items()]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.info("No supervised model was trained (no labelled history).")

    with tabs[2]:
        st.subheader("Priority cases")
        cases = pd.read_csv(output / "flagged_cases.csv")
        if cases.empty:
            st.success("Nothing flagged.")
        else:
            st.dataframe(cases, width="stretch", hide_index=True)

    with tabs[3]:
        st.subheader("Live batch score table")
        scored = result["scored"]
        show = scored[["txn_datetime", "txn_id", "account_id", "merchant",
                       "merchant_category", "country", "channel", "amount",
                       "model_proba", "anomaly_score", "rule_score",
                       "risk_score", "severity", "disposition", "flags",
                       "reason"]]
        st.dataframe(show, width="stretch", hide_index=True)
        st.caption("`reason` records the evidence behind every decision "
                   "(the audit trail).")

    with tabs[4]:
        st.subheader("Agent reasoning trail")
        for step in trail:
            st.markdown(f"**[{step['step']}] {step['action']}** — {step['detail']}")

    with tabs[5]:
        st.subheader("Downloads")
        for p in sorted(output.glob("*.*")):
            if p.is_file() and p.suffix in (".csv", ".json", ".html"):
                st.download_button(
                    f"Download {p.name}",
                    p.read_bytes(),
                    file_name=p.name,
                    mime="text/csv" if p.suffix == ".csv"
                    else "application/json" if p.suffix == ".json"
                    else "text/html",
                )

    st.divider()
    st.caption(
        "Pipeline: LOAD - PROFILE - FEATURE - LEARN - SCORE - DECIDE - "
        "EVALUATE - AUDIT - ANALYSE - OUTPUT. Scores blend a Random Forest "
        "(65%) + Isolation Forest anomaly score (20%) + rule engine (15%). "
        "Alerts are decision support only - investigate before acting."
    )


if __name__ == "__main__":
    main()