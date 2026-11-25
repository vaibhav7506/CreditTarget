"""Vectorized PySpark pandas_udf NLP complaint, sentiment, and distress scorer."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from pyspark.sql.functions import pandas_udf

from credittarget.schemas.contracts import NLP_RESULT_SCHEMA

# Domain-specific banking NLP lexicons
POSITIVE_TERMS = {
    "reward",
    "rewards",
    "points",
    "travel",
    "increase",
    "good",
    "happy",
    "perks",
    "upgrade",
    "premier",
}
NEGATIVE_TERMS = {
    "dispute",
    "unacceptable",
    "hardship",
    "emergency",
    "close",
    "unsatisfactory",
    "fees",
    "complaint",
    "refuse",
    "terrible",
}
DISTRESS_TERMS = {
    "hardship",
    "medical emergency",
    "unable to pay",
    "unemployed",
    "missed payment",
    "debt relief",
    "can't pay",
}
COMPLAINT_TERMS = {
    "dispute",
    "unacceptable",
    "demanding refund",
    "close account",
    "regulator",
    "lawyer",
    "fraud",
    "complaint",
}
CREDIT_INTENT_TERMS = {
    "credit limit increase",
    "limit increase",
    "higher limit",
    "loan offer",
    "card offer",
    "travel perks",
    "reward points",
}


def score_single_note_text(text: Any) -> dict[str, float | int]:
    """Analyzes a single support note string with negation awareness."""
    if not isinstance(text, str) or not text.strip():
        return {
            "sentiment_score": 0.0,
            "distress_flag": 0,
            "complaint_severity": 0.0,
            "credit_intent_flag": 0,
        }

    raw = text.lower().strip()
    words = re.findall(r"\b\w+\b", raw)

    # Check for negation phrases like "no hardship", "not happy", "no dispute"
    has_no_hardship = bool(
        re.search(r"\b(?:no|not|without|never)\s+(?:financial\s+)?hardship\b", raw)
    )
    has_not_happy = bool(re.search(r"\b(?:not|never)\s+happy\b", raw))

    # Sentiment scoring
    pos_count = sum(1 for w in words if w in POSITIVE_TERMS)
    neg_count = sum(1 for w in words if w in NEGATIVE_TERMS)
    if has_not_happy:
        pos_count = max(0, pos_count - 1)
        neg_count += 1

    len(words) or 1
    raw_sentiment = (pos_count - neg_count) / max(1.0, float(pos_count + neg_count))
    sentiment = max(-1.0, min(1.0, round(raw_sentiment, 4)))

    # Distress detection
    distress = 0
    if not has_no_hardship:
        for phrase in DISTRESS_TERMS:
            if phrase in raw:
                distress = 1
                break

    # Complaint severity [0.0, 1.0]
    severity = 0.0
    for term in COMPLAINT_TERMS:
        if term in raw:
            severity += 0.35
    if "unacceptable" in raw:
        severity += 0.25
    if "close account" in raw or "demanding refund" in raw:
        severity += 0.25
    complaint_severity = min(1.0, round(severity, 4))

    # Credit intent detection
    credit_intent = 0
    for term in CREDIT_INTENT_TERMS:
        if term in raw:
            credit_intent = 1
            break

    return {
        "sentiment_score": sentiment,
        "distress_flag": distress,
        "complaint_severity": complaint_severity,
        "credit_intent_flag": credit_intent,
    }


def score_support_notes_pandas(series: pd.Series) -> pd.DataFrame:
    """Vectorized pandas scorer returning a DataFrame matching NLP_RESULT_SCHEMA."""
    results = [score_single_note_text(t) for t in series]
    df = pd.DataFrame(results)
    df["sentiment_score"] = df["sentiment_score"].astype("float64")
    df["distress_flag"] = df["distress_flag"].astype("int32")
    df["complaint_severity"] = df["complaint_severity"].astype("float64")
    df["credit_intent_flag"] = df["credit_intent_flag"].astype("int32")
    return df


@pandas_udf(NLP_RESULT_SCHEMA)  # type: ignore[call-overload, untyped-decorator]
def get_nlp_scorer_udf(text_series: pd.Series) -> pd.DataFrame:
    """PySpark vectorized pandas_udf for distributed execution."""
    return score_support_notes_pandas(text_series)
