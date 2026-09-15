# Bank Reconciliation Agent

An **AI agent for accounting** that automates the monthly **bank-to-ledger reconciliation**
process: it matches bank statement transactions to general-ledger (GL) entries, flags unmatched
and suspicious items, ages outstanding transactions and produces an auditable reconciliation
report — freeing the accountant to focus on the handful of items that actually need judgement.

**Project 2 · AI Agent for Accounting** — a Streamlit web application that accepts accounting data
(bank statement + ledger), runs the agent in the browser, and presents its findings and recommendations.

---

## 1. Business problem

Reconciliation is one of the most manual, repetitive and error-prone tasks in the accounting
close process. Every month an accountant must prove that the cash reflected in the bank
statement equals the cash recorded in the GL cash book. In practice:

- The same transaction is booked on **different days** by the bank and the ledger.
- Descriptions differ (e.g. `EFT Rent payment` vs `Rent deposit`).
- Amounts differ by small amounts ($0.06 rounding, bank fees, currency conversions).
- One payment is **split** across several ledger lines.
- Genuinely unmatched items (bank charges not yet posted, unbanked cash, uncleared cheques,
  duplicated postings) hide in hundreds of lookups and spreadsheet `VLOOKUP`s.

**Why it matters:** unmatched items left on the reconciliation file distort the cash balance,
delay the close, hide control failures, and can mask fraudulent or duplicated payments.

**Why an agent:** a staged, transparent process (load → match → flag → age → report) with a
visible reasoning trail gives the accountant **decision support, not a black box**. The agent
mimics how a meticulous accountant reconciles, but runs in seconds and never misses a candidate.

## 2. How the agent works (methodology)

The agent is a **rule-and-statistics hybrid**, implemented in pure Python (Decimal arithmetic to
avoid floating-point errors, zero external model dependencies, no data leaves the process).

| Stage | What the agent does |
|---|---|
| 1. Load & normalise | Parses CSV/XLSX bank and ledger files; tolerates varied column names (`date`, `posting_date`, `txn_date`, ...) and formats, including separate `debit`/`credit` columns. |
| 2. Pass 1 — exact match | Pairs transactions with identical date **and** amount. |
| 3. Pass 2 — date-window match | Pairs equal amounts recorded within a configurable window (default 5 days). |
| 4. Pass 3 — split match | Finds groups of 2–4 lines whose amounts sum exactly to a single counterpart (e.g. one rent EFT split across two ledger postings). |
| 5. Pass 4 — fuzzy match | Pairs nearly-equal amounts within a configurable tolerance (default 0.1%), within the date window. |
| 6. Summarise | Computes match rate, matched/unmatched values and a **reconciliation difference**. |
| 7. Age | Buckets unmatched bank items into 0–30, 31–60, 61–90 and >90 days. |
| 8. Assess | Generates **actionable alerts**: unposted bank charges (posting recommendation), large round-number deposits, possible duplicate postings, first-digit (**Benford's law**) deviation, aged items. |
| 9. Report | Produces a human-readable report, a structured JSON export and a CSV of matched / unmatched items. |

Every stage writes to an **agent trail** shown in the app, so each match is auditable
(which pass matched it, and why).

## 3. Sample datasets

`sample_data/` contains fictional, anonymised data for a small trading business over
**June–July 2026** with deliberately seeded problems so every capability can be demonstrated:

| Item | What it demonstrates |
|---|---|
| Bank service charge $12.50 | Unposted bank charge → **posting recommendation** |
| $50,000.00 round deposit | Unrecorded deposit → **round-number anomaly** |
| Supplier A $1,234.56 vs $1,234.50 | **Fuzzy match** (0.06 difference) |
| Services fee (2-day gap) / payroll (gaps) | **Date-window matching** |
| Rent EFT $1,800 vs $1,000 + $800 | **Split matching** |
| Two identical payroll lines | **Duplicate detection** |
| Cash-not-banked, uncleared cheque | Ledger-only items → investigate |

Expected result on the sample: **19 of 22** bank lines matched (**86.4%**), 3 unmatched on each
side, and **10 alerts**.

Regenerate the datasets anytime:

```bash
python scripts/generate_sample_data.py
```

## 4. Repository layout

```
bank-reconciliation-agent/
├── app.py                        # Streamlit web application (SELF-CONTAINED)
├── reconcile/
│   ├── __init__.py
│   └── engine.py                 # The agent: parsing, matching, anomaly, ageing, report (canonical)
├── sample_data/
│   ├── bank_statement.csv        # Simulated bank statement
│   └── ledger.csv                # Simulated GL cash book
├── notebooks/
│   └── Bank_Reconciliation_Agent.ipynb   # Standalone Colab notebook (no setup)
├── scripts/
│   ├── generate_sample_data.py   # Regenerates sample_data/
│   ├── build_notebook.py         # Rebuilds the Colab notebook from engine + data
│   └── sync_app.py               # Re-embeds engine + sample data into app.py
├── tests/
│   ├── test_reconcile.py         # 17 unit tests (parser, matching passes, anomalies, ageing)
│   └── test_parity.py            # Ensures the embedded engine == reconcile engine (drift guard)
├── requirements.txt
├── .streamlit/config.toml        # Theme + headless config
└── README.md
```

> **Why `app.py` is self-contained:** the engine and sample datasets are *embedded* in
> `app.py` so the deployed Streamlit app runs even if only `app.py` reaches the repository.
> `reconcile/engine.py` remains the canonical library used by the tests and notebook;
> `scripts/sync_app.py` regenerates the embedded copies, and `tests/test_parity.py` verifies
> the two never drift apart (`python scripts/sync_app.py` after any engine change).

## 5. Run it locally

```bash
# 1. Create a virtual environment (optional but recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Open `http://localhost:8501`, choose **Simulated sample dataset** (or **Upload files** with your
own CSV/XLSX), and press **Run reconciliation agent**.

Run the tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

After any change to `reconcile/engine.py` or `sample_data/`, keep the deployed app in sync:

```bash
python scripts/sync_app.py
```

## 6. Run the Colab notebook

Open [`notebooks/Bank_Reconciliation_Agent.ipynb`](notebooks/Bank_Reconciliation_Agent.ipynb)
in [Google Colab](https://colab.research.google.com) and run all cells. The engine and sample
data are embedded in the notebook, so it runs with **zero setup** (standard library only).

## 7. Deploy app to Streamlit Community Cloud

1. Push this folder to a GitHub repository (see below).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select the repository, branch, and set **Main file path** to `app.py`.
4. Deploy. The app rebuilds from the repo — keep `app.py` at the repository root.

> **Deploy tip:** `app.py` is fully self-contained (engine + sample data embedded), so the app will
> run even if the surrounding package folders are not in the repository. For full marks, still push
> **all** files — source, tests, datasets, notebook and README — as required by the assignment.

### Push to GitHub (from this folder)

```bash
git init
git add -A
git commit -m "Bank Reconciliation Agent - AI agent for accounting"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

## 8. Security & confidentiality

- **No API keys or passwords are used.** The agent is fully local: matching, anomaly scoring and
  reporting all run in the browser session; uploaded data is never sent to any external service.
- **No confidential data is committed.** The repository ships only simulated, anonymised data.
- If the app is ever extended with external services, store credentials in
  `.streamlit/secrets.toml` (which is **git-ignored** in this repository) or Streamlit Cloud
  Secrets — never commit them.

## 9. Testing & evaluation

17 unit tests cover: amount/date parsing (including `debit`/`credit` columns and alias columns),
each matching pass, no-overlap guarantees, reconciled balance algebra, anomaly alerts, ageing
buckets, and parameter sensitivity. Evaluation metrics surfaced in the app for the sample run:

- **Match rate:** 86.4% of bank lines (19/22)
- **Unmatched bank:** 3 lines ($50,767.50) · **Unmatched ledger:** 3 lines (net −$6,549.00)
- **Reconciliation difference:** $44,218.50 (outstanding items on both sides)
- **Alerts:** 10 (posting recommendation, round-number anomaly, duplicate posting, aged items, unmatched items)

## 10. Limitations

- Deterministic rule-based matching: very unusual layouts or non-standard date formats may need
  a small column-alias addition.
- Split matching is limited to exact-sum combinations of up to 4 lines.
- Benford's law is a heuristic aid, not proof of fraud.
- Advisory only: the agent recommends, the accountant decides.

## 11. License

Educational project; provided for academic submission. All sample data is fictional.