"""
Fraud Detection Agent - core engine.

A pure-Python, rule-and-statistics based screening pipeline for a finance team
that processes an accounts-payable transaction file. It:

  1. LOAD      - parses and normalises transaction rows (CSV/TSV/XLSX text).
  2. ANALYSE   - runs detection modules: duplicate payments, sub-threshold
                 (just-below-limit) amounts, large round numbers, high-value
                 outliers, vendor velocity, suspicious descriptions, new /
                 one-off vendors and weekend activity.
  3. DECIDE    - scores every transaction 0-100 from the active flags and
                 assigns a severity band (LOW / MEDIUM / HIGH / CRITICAL).
  4. AGGREGATE - runs dataset-level checks (Benford's first-digit law, vendor
                 velocity bursts, sub-threshold stacking) and builds a
                 prioritised case list.
  5. OUTPUT    - produces a human-readable report, a structured JSON export and
                 a fully explained ledger (one reason per flag per row).

No external services are used: no data ever leaves the process, which makes the
agent appropriate for confidential payment data. Amounts are handled as Decimal
to avoid floating point drift.
"""
from __future__ import annotations

import csv
import io
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Column name aliases so the agent tolerates real-world exports.
# ---------------------------------------------------------------------------
DATE_COLS = ("date", "posting_date", "txn_date", "transaction_date", "value_date", "payment_date")
DESC_COLS = ("description", "particulars", "narration", "memo", "details", "transaction", "remarks")
AMOUNT_COLS = ("amount", "value", "amount_orig", "net_amount", "paid")
REF_COLS = ("reference", "ref", "transaction_id", "doc_number", "invoice_no", "cheque_no")
PAYEE_COLS = ("payee", "vendor", "supplier", "counterparty", "beneficiary", "merchant", "creditor")
CATEGORY_COLS = ("category", "expense_category", "type", "class", "account")
DEBIT_COLS = ("debit", "withdrawal", "money_out")
CREDIT_COLS = ("credit", "deposit", "money_in")

D2 = Decimal("0.01")


class ParseError(ValueError):
    """Raised when a supplied file cannot be parsed."""


def parse_amount(value) -> Decimal:
    """Parse a value into a Decimal, tolerating commas, currency symbols and
    parenthesised negatives."""
    if isinstance(value, Decimal):
        return value.quantize(D2, rounding=ROUND_HALF_UP)
    if value is None:
        raise ParseError("empty amount")
    s = str(value).strip()
    if s == "":
        raise ParseError("empty amount")
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    for ch in "$,RZWL\u00a0":
        s = s.replace(ch, "")
    s = s.strip()
    try:
        d = Decimal(s)
    except InvalidOperation:
        raise ParseError(f"cannot parse amount: {value!r}")
    if negative:
        d = -d
    return d.quantize(D2, rounding=ROUND_HALF_UP)


def _parse_date(value) -> date:
    if isinstance(value, date):
        return value
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d %b %Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    try:
        return date.fromisoformat(s)
    except ValueError:
        raise ParseError(f"cannot parse date: {value!r}")


def _first(row: Dict[str, str], names: Sequence[str]) -> Optional[str]:
    lowered = {k.lower().strip(): v for k, v in row.items()}
    for name in names:
        if name in lowered and (lowered[name] or "").strip():
            return lowered[name]
    return None


def parse_transactions_csv(text: str, source: str = "payment") -> List["Transaction"]:
    """Parse a CSV/TSV string into Transaction objects.

    Tolerates common column layouts, including separate `debit` / `credit`
    columns instead of a signed amount. Leading blank lines are ignored.
    """
    text = text.lstrip("\ufeff \t\n\r")
    if not text:
        raise ParseError("file is empty")
    if "\t" in text and "," not in text.splitlines()[0]:
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    else:
        reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ParseError("file has no header row")
    rows: List[Transaction] = []
    for i, row in enumerate(reader, start=1):
        if not any((v or "").strip() for v in row.values()):
            continue
        try:
            rows.append(Transaction.from_row(row, source=source))
        except ParseError as exc:
            raise ParseError(f"row {i}: {exc}")
    if not rows:
        raise ParseError("no transaction rows found")
    return rows


@dataclass
class Flag:
    """One detected evidence item attached to a transaction."""
    code: str
    label: str
    points: int
    detail: str

    def as_dict(self) -> Dict[str, str]:
        return {"code": self.code, "label": self.label, "points": self.points, "detail": self.detail}


@dataclass
class Transaction:
    """A single payment / transfer row being screened."""
    date: date
    amount: Decimal
    description: str
    reference: str = ""
    payee: str = ""
    category: str = ""
    source: str = "payment"
    flags: List[Flag] = field(default_factory=list)
    score: int = 0
    severity: str = "LOW"

    @classmethod
    def from_row(cls, row: Dict[str, str], source: str) -> "Transaction":
        date_raw = _first(row, DATE_COLS)
        desc_raw = _first(row, DESC_COLS)
        ref_raw = _first(row, REF_COLS)
        payee_raw = _first(row, PAYEE_COLS)
        cat_raw = _first(row, CATEGORY_COLS)
        amount_raw = _first(row, AMOUNT_COLS)
        if amount_raw is None:
            debit = _first(row, DEBIT_COLS)
            credit = _first(row, CREDIT_COLS)
            if debit is None and credit is None:
                raise ParseError("no amount or debit/credit columns found")
            debit = parse_amount(debit or 0)
            credit = parse_amount(credit or 0)
            amount = credit - debit
        else:
            amount = parse_amount(amount_raw)
        if date_raw is None:
            raise ParseError("no date column found")
        return cls(
            date=_parse_date(date_raw),
            amount=amount,
            description=(desc_raw or "").strip(),
            reference=(ref_raw or "").strip(),
            payee=(payee_raw or "").strip(),
            category=(cat_raw or "").strip(),
            source=source,
        )

    def recipient(self) -> str:
        return self.payee or self.category or (self.description or "unknown")

    def as_dict(self) -> Dict[str, str]:
        return {
            "date": self.date.isoformat(),
            "reference": self.reference,
            "description": self.description,
            "payee": self.payee,
            "category": self.category,
            "amount": str(self.amount),
            "score": str(self.score),
            "severity": self.severity,
            "flags": ", ".join(f.code for f in self.flags) or "-",
            "explain": "; ".join(f.detail for f in self.flags) or "No anomalies",
        }


@dataclass
class Alert:
    """An actionable insight produced by the agent."""
    severity: str          # HIGH | MEDIUM | LOW
    kind: str              # e.g. DUPLICATE | JUST_BELOW | BENFORD | VELOCITY
    text: str

    def as_dict(self) -> Dict[str, str]:
        return {"severity": self.severity, "kind": self.kind, "text": self.text}


@dataclass
class Case:
    """A prioritized case for the investigator to review."""
    reference: str
    date: str
    recipient: str
    amount: str
    score: int
    severity: str
    top_flags: str
    explain: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "reference": self.reference,
            "date": self.date,
            "recipient": self.recipient,
            "amount": self.amount,
            "score": self.score,
            "severity": self.severity,
            "flags": self.top_flags,
            "explain": self.explain,
        }


@dataclass
class FraudResult:
    """Complete output of the screening pipeline."""
    transactions: List[Transaction] = field(default_factory=list)
    alerts: List[Alert] = field(default_factory=list)
    cases: List[Case] = field(default_factory=list)
    trail: List[Dict] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)

    def flagged(self) -> List[Transaction]:
        return [t for t in self.transactions if t.severity in ("MEDIUM", "HIGH", "CRITICAL")]

    def to_dict(self) -> Dict:
        return {
            "summary": self.summary,
            "transactions": [t.as_dict() for t in self.transactions],
            "cases": [c.as_dict() for c in self.cases],
            "alerts": [a.as_dict() for a in self.alerts],
            "trail": self.trail,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def report_text(self) -> str:
        """Human-readable fraud screening report (agent write-up)."""
        lines = ["=" * 72, "FRAUD DETECTION AGENT - SCREENING REPORT", "=" * 72]
        s = self.summary
        lines.append("")
        lines.append(f"Period analysed          : {s.get('period')}")
        lines.append(f"Transactions screened    : {s.get('total', 0)}  ({s.get('total_amount')})")
        lines.append(f"Flagged for review       : {s.get('flagged', 0)}  ({s.get('flagged_amount')})")
        lines.append(f"Critical / High / Medium : {s.get('critical', 0)} / {s.get('high', 0)} / {s.get('medium', 0)}")
        lines.append(f"Open cases               : {s.get('cases', 0)}")
        lines.append("")
        lines.append("-" * 72)
        lines.append("AGENT TRAIL")
        lines.append("-" * 72)
        for step in self.trail:
            lines.append(f"[{step['step']}] {step['action']}: {step['detail']}")
        lines.append("")
        lines.append("-" * 72)
        lines.append("ALERTS")
        lines.append("-" * 72)
        for a in self.alerts:
            lines.append(f"[{a.severity}] ({a.kind}) {a.text}")
        lines.append("")
        lines.append("-" * 72)
        lines.append("PRIORITISED CASES")
        lines.append("-" * 72)
        for c in self.cases:
            lines.append(f"{c.severity:>8} risk {c.score:>3}  {c.date}  {c.reference:<12} {c.amount:>12}  {c.recipient}")
            lines.append(f"          flags: {c.top_flags}")
            lines.append(f"          why  : {c.explain}")
        lines.append("")
        lines.append("-" * 72)
        lines.append("ALL SCREENED ROWS")
        lines.append("-" * 72)
        for t in sorted(self.transactions, key=lambda x: (-x.score, x.date)):
            lines.append(f"{t.severity:>8} risk {t.score:>3}  {t.date.isoformat()}  {t.amount:>12}  {t.recipient()}")
        lines.append("")
        lines.append("=" * 72)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class FraudConfig:
    """Tunable policy thresholds for the detection modules."""
    approval_limit: Decimal = Decimal("10000")   # amounts under this escape authorisation
    just_below_pct: Decimal = Decimal("0.05")    # amounts within 5% below a limit are suspicious
    duplicate_window_days: int = 7               # span for duplicate detection (recurring rent gap > 30d)
    outlier_factor: Decimal = Decimal("10")       # vs median absolute amount
    outlier_min: Decimal = Decimal("5000")       # absolute floor for outlier flag
    round_min: Decimal = Decimal("5000")         # round numbers >= this are flagged
    round_step: Decimal = Decimal("1000")        # exact multiples of this
    velocity_count: int = 3                      # >= this many payments to one recipient
    velocity_window_days: int = 14               # within this span
    velocity_min_total: Decimal = Decimal("10000")  # combined value floor for a burst
    new_vendor_min: Decimal = Decimal("5000")    # one-time recipient >= this is flagged
    benford_min_sample: int = 15                 # first-digit test needs this many values
    benford_mad_threshold: float = 0.08          # mean absolute deviation threshold
    suspicious_keywords: Tuple[str, ...] = (
        "misc", "adjustment", "refund", "write-off", "writeoff", "urgent",
        "cash", "reimburs", "transfer", "correction", "error", "void",
    )
    severity_bands: Tuple[Tuple[int, str], ...] = (
        (100, "CRITICAL"),
        (70, "HIGH"),
        (40, "MEDIUM"),
        (0, "LOW"),
    )


# ---------------------------------------------------------------------------
# Detection modules
# ---------------------------------------------------------------------------
def _severity_for(score: int, config: FraudConfig) -> str:
    for threshold, label in config.severity_bands:
        if score >= threshold:
            return label
    return "LOW"


def _first_digit_dist(values: Iterable[Decimal]) -> Dict[int, int]:
    counts = {d: 0 for d in range(1, 10)}
    for v in values:
        try:
            if v <= 0:
                continue
            s = str(v.quantize(D2)).lstrip("-0") or "0"
            if not s or s[0] in ".0":
                continue
            digit = int(s[0])
            if 1 <= digit <= 9:
                counts[digit] += 1
        except Exception:
            continue
    return counts


def _benford_mad(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    deviations = 0.0
    for d in range(1, 10):
        observed = counts[d] / total
        expected = math.log10(1 + 1 / d)
        deviations += abs(observed - expected)
    return deviations / 9.0


def _detect_rows(rows: Sequence[Transaction], config: FraudConfig) -> None:
    """Apply rule-based flags to each transaction in place."""
    med = median([abs(t.amount) for t in rows]) if rows else Decimal("0")

    dup_key = defaultdict(list)
    for t in rows:
        key = (t.recipient().lower(), t.amount)
        dup_key[key].append(t)

    payee_maps: Dict[str, List[Transaction]] = defaultdict(list)
    for t in rows:
        payee_maps[t.recipient().lower()].append(t)

    limits = {config.approval_limit, config.approval_limit * Decimal("0.5"),
              config.approval_limit * Decimal("2"), config.approval_limit * Decimal("5")}

    for t in rows:
        amt = abs(t.amount)

        # Duplicate payment (same recipient, same amount, close together).
        members = dup_key.get((t.recipient().lower(), t.amount), [])
        if len(members) >= 2:
            dates = [m.date for m in members]
            if (max(dates) - min(dates)).days <= config.duplicate_window_days:
                t.flags.append(Flag("DUPLICATE", "Possible duplicate payment", 45,
                                    f"same recipient ({t.recipient()}) and amount {t.amount} "
                                    f"paid {len(members)}x within {config.duplicate_window_days} days"))

        # Just-below an authorisation limit (sub-threshold evasion).
        for lim in limits:
            if amt < lim and lim - amt <= lim * config.just_below_pct:
                t.flags.append(Flag("JUST_BELOW", "Just below an approval limit", 40,
                                    f"{t.amount} sits {lim - amt:g} below the {lim:g} authorisation limit"))
                break

        # Large exact round number (fabricated-invoice red flag).
        if amt >= config.round_min and amt % config.round_step == 0:
            t.flags.append(Flag("ROUND", "Large round-number amount", 35,
                                f"{t.amount} is an exact multiple of {config.round_step}"))

        # High-value outlier vs the median of the file.
        if amt >= config.outlier_min and amt > med * config.outlier_factor:
            t.flags.append(Flag("OUTLIER", "High-value outlier", 65,
                                f"{t.amount} is {amt / med if med else 0:.1f}x the median payment ({med})"))

        # Suspicious wording in the narration.
        lowered = t.description.lower()
        hits = [k for k in config.suspicious_keywords if k in lowered]
        if hits:
            t.flags.append(Flag("KEYWORD", "Suspicious description keyword", 15,
                                f"narration contains {', '.join(hits)}"))

        # Weekend posting.
        if t.date.weekday() >= 5:
            t.flags.append(Flag("WEEKEND", "Weekend posting", 5,
                                f"{t.date.isoformat()} is a weekend date"))

    # Vendor velocity (many payments to one recipient in a short span).
    for key, members in payee_maps.items():
        if len(members) < config.velocity_count:
            continue
        dates = sorted(m.date for m in members)
        span = (dates[-1] - dates[0]).days
        if span > config.velocity_window_days:
            continue
        total = sum((abs(m.amount) for m in members), Decimal("0"))
        if total < config.velocity_min_total:
            continue
        for t in members:
            t.flags.append(Flag("VELOCITY", "High payment velocity to one recipient", 40,
                                f"{len(members)} payments totalling {total} to {t.recipient()} "
                                f"within {span} days"))

    # One-off recipient with a material amount.
    for key, members in payee_maps.items():
        if len(members) == 1 and abs(members[0].amount) >= config.new_vendor_min:
            t = members[0]
            t.flags.append(Flag("NEW_VENDOR", "One-off recipient with material value", 20,
                                f"{t.recipient()} appears only once with {t.amount}"))


def _score_rows(rows: Sequence[Transaction], config: FraudConfig) -> None:
    for t in rows:
        seen = set()
        points = 0
        for flag in t.flags:
            if flag.code in seen:
                continue
            seen.add(flag.code)
            points += flag.points
        t.score = min(100, points)
        t.severity = _severity_for(t.score, config)


def _build_alerts(rows: Sequence[Transaction], config: FraudConfig) -> List[Alert]:
    alerts: List[Alert] = []

    dup_groups = defaultdict(list)
    for t in rows:
        dup_groups[(t.recipient().lower(), t.amount)].append(t)
    for key, members in dup_groups.items():
        if len(members) >= 2:
            dates = sorted(m.date for m in members)
            if (dates[-1] - dates[0]).days <= config.duplicate_window_days:
                estr = ", ".join(f"{m.date.isoformat()} ({m.amount})" for m in members)
                alerts.append(Alert("HIGH", "DUPLICATE",
                                    f"Possible duplicate payment to {members[0].recipient()} "
                                    f"({members[0].amount}) paid {len(members)}x: {estr} - verify the invoices before releasing."))

    sub_thresh = defaultdict(list)
    for t in rows:
        amt = abs(t.amount)
        for lim in {config.approval_limit, config.approval_limit * Decimal("0.5"), config.approval_limit * Decimal("2")}:
            if amt < lim and lim - amt <= lim * config.just_below_pct:
                sub_thresh[t.recipient().lower()].append(t)
                break
    for key, members in sub_thresh.items():
        if sum((1 for m in members), 0) >= 2:
            total = sum((abs(m.amount) for m in members), Decimal("0"))
            alerts.append(Alert("HIGH", "JUST_BELOW",
                                f"{len(members)} payments to {members[0].recipient()} sit just below an "
                                f"authorisation limit (total {total}) - possible invoice splitting."))
        elif members:
            m = members[0]
            alerts.append(Alert("MEDIUM", "JUST_BELOW",
                                f"{m.amount} to {m.recipient()} on {m.date.isoformat()} sits just below an "
                                f"authorisation limit - confirm approval evidence."))

    payee_maps: Dict[str, List[Transaction]] = defaultdict(list)
    for t in rows:
        payee_maps[t.recipient().lower()].append(t)
    for key, members in payee_maps.items():
        if len(members) < config.velocity_count:
            continue
        dates = sorted(m.date for m in members)
        span = (dates[-1] - dates[0]).days
        if span <= config.velocity_window_days:
            total = sum((abs(m.amount) for m in members), Decimal("0"))
            if total >= config.velocity_min_total:
                alerts.append(Alert("MEDIUM", "VELOCITY",
                                    f"{len(members)} payments totalling {total} to {members[0].recipient()} "
                                    f"within {span} days - unusual concentration."))

    digits = _first_digit_dist(t.amount for t in rows if t.amount > 0)
    if sum(digits.values()) >= config.benford_min_sample:
        mad = _benford_mad(digits)
        if mad > config.benford_mad_threshold:
            alerts.append(Alert("MEDIUM", "BENFORD",
                                f"First-digit distribution of payment amounts deviates from Benford's law "
                                f"(MAD {mad:.3f}) - the dataset may contain fabricated amounts."))

    weekend = [t for t in rows if t.date.weekday() >= 5]
    if weekend:
        total = sum((abs(t.amount) for t in weekend), Decimal("0"))
        alerts.append(Alert("LOW", "WEEKEND",
                            f"{len(weekend)} payment(s) totalling {total} posted on a weekend - "
                            f"confirm authorisation and review logs."))

    return alerts


def _build_cases(rows: Sequence[Transaction], limit: int = 20) -> List[Case]:
    flagged = [t for t in rows if t.severity in ("MEDIUM", "HIGH", "CRITICAL")]
    flagged.sort(key=lambda x: (-x.score, x.date))
    cases: List[Case] = []
    for t in flagged[:limit]:
        cases.append(Case(
            reference=t.reference or "-",
            date=t.date.isoformat(),
            recipient=t.recipient(),
            amount=str(t.amount),
            score=t.score,
            severity=t.severity,
            top_flags=", ".join(f.code for f in t.flags) or "-",
            explain="; ".join(f.detail for f in t.flags) or "No anomalies",
        ))
    return cases


def _build_summary(rows: Sequence[Transaction], alerts: Sequence[Alert], cases: Sequence[Case]) -> Dict:
    total = len(rows)
    total_amount = sum((t.amount for t in rows), Decimal("0"))
    flagged = [t for t in rows if t.severity in ("MEDIUM", "HIGH", "CRITICAL")]
    flagged_amount = sum((t.amount for t in flagged), Decimal("0"))
    counts = Counter(t.severity for t in rows)
    all_dates = [t.date for t in rows]
    period = ""
    if all_dates:
        period = f"{min(all_dates).isoformat()} to {max(all_dates).isoformat()}"
    return {
        "period": period,
        "total": total,
        "total_amount": str(total_amount),
        "flagged": len(flagged),
        "flagged_amount": str(flagged_amount),
        "critical": counts["CRITICAL"],
        "high": counts["HIGH"],
        "medium": counts["MEDIUM"],
        "cases": len(cases),
        "alerts": len(alerts),
        "flagged_pct": round(len(flagged) / total * 100, 1) if total else 0.0,
    }


# ---------------------------------------------------------------------------
# Agent pipeline
# ---------------------------------------------------------------------------
def screen(rows: Sequence[Transaction], config: Optional[FraudConfig] = None) -> FraudResult:
    """Run the full fraud screening agent pipeline."""
    config = config or FraudConfig()
    rows = list(rows)
    result = FraudResult(transactions=rows)
    result.trail.append({"step": 1, "action": "Load", "detail": f"{len(rows)} transaction(s) parsed and normalised"})

    _detect_rows(rows, config)
    result.trail.append({"step": 2, "action": "Analyse",
                         "detail": f"{sum((len(t.flags) for t in rows), 0)} evidence flag(s) raised by the detection modules"})

    _score_rows(rows, config)
    result.trail.append({"step": 3, "action": "Decide",
                         "detail": f"every transaction scored 0-100 and assigned a severity band"})

    result.alerts = _build_alerts(rows, config)
    result.trail.append({"step": 4, "action": "Aggregate",
                         "detail": f"{len(result.alerts)} actionable alert(s) generated (duplicates, sub-threshold, velocity, Benford, weekend)"})

    result.cases = _build_cases(rows)
    result.trail.append({"step": 5, "action": "Prioritise",
                         "detail": f"{len(result.cases)} case(s) ranked by risk score for review"})

    result.summary = _build_summary(rows, result.alerts, result.cases)
    result.trail.append({"step": 6, "action": "Complete", "detail": "screening ready for review"})

    return result