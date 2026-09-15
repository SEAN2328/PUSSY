"""Fraud Detection Agent package.

A finance-fraud screening agent that combines an explainable rule-and-statistics
engine with machine-learning anomaly detection and supervised classification.

Pipeline (see ``frauddetect.agent.FraudAgent.run``):

    LOAD  -> PROFILE -> FEATURE -> LEARN (optional) -> SCORE
    -> DECIDE -> EVALUATE (optional) -> AUDIT -> ANALYSE -> OUTPUT
"""
from .agent import AgentConfig, FraudAgent, FraudAgentError
from .engine import (
    Alert,
    Case,
    Flag,
    FraudConfig,
    FraudResult,
    ParseError,
    Transaction,
    parse_amount,
    parse_transactions_csv,
    screen,
)

__all__ = [
    "Alert",
    "AgentConfig",
    "Case",
    "Flag",
    "FraudAgent",
    "FraudAgentError",
    "FraudConfig",
    "FraudResult",
    "ParseError",
    "Transaction",
    "parse_amount",
    "parse_transactions_csv",
    "screen",
]