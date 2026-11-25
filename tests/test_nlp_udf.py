"""Tests for NLP scoring UDF: sentiment, distress, complaint severity, credit intent, null handling, and negation."""

from __future__ import annotations

import pandas as pd

from credittarget.pipelines.nlp_udf import score_single_note_text, score_support_notes_pandas


def test_nlp_scorer_distress_detection() -> None:
    """Verify distress flag triggers on financial hardship terms."""
    distress_text = (
        "Hardship notice: customer experienced medical emergency, unable to pay minimum due on time"
    )
    res = score_single_note_text(distress_text)
    assert res["distress_flag"] == 1
    assert res["sentiment_score"] < 0.0


def test_nlp_scorer_negation_awareness() -> None:
    """Verify negation handling for 'no hardship' and 'not happy'."""
    no_hardship_text = "Customer confirmed they have no hardship and are paying the balance today"
    res = score_single_note_text(no_hardship_text)
    assert res["distress_flag"] == 0

    not_happy_text = "Customer is not happy with the service"
    res_neg = score_single_note_text(not_happy_text)
    assert res_neg["sentiment_score"] < 0.0


def test_nlp_scorer_credit_intent() -> None:
    """Verify credit intent detection."""
    intent_text = "Customer requested credit limit increase for international travel"
    res = score_single_note_text(intent_text)
    assert res["credit_intent_flag"] == 1
    assert res["sentiment_score"] > 0.0


def test_nlp_scorer_complaint_severity() -> None:
    """Verify complaint severity scale."""
    complaint_text = "Customer dispute: billing fee charged is completely unacceptable, demanding refund or will close account"
    res = score_single_note_text(complaint_text)
    assert res["complaint_severity"] >= 0.80


def test_nlp_scorer_empty_and_null() -> None:
    """Verify graceful handling of empty, None, and NaN inputs."""
    res_none = score_single_note_text(None)
    assert res_none == {
        "sentiment_score": 0.0,
        "distress_flag": 0,
        "complaint_severity": 0.0,
        "credit_intent_flag": 0,
    }

    res_empty = score_single_note_text("")
    assert res_empty == {
        "sentiment_score": 0.0,
        "distress_flag": 0,
        "complaint_severity": 0.0,
        "credit_intent_flag": 0,
    }


def test_nlp_scorer_vectorized_pandas() -> None:
    """Verify vectorized pandas Series processing matches expected types."""
    series = pd.Series(
        [
            "Great rewards points on my premier card!",
            "Dispute billing charge unacceptable close account",
            None,
            "Hardship notice medical emergency cannot pay",
        ]
    )
    df = score_support_notes_pandas(series)
    assert len(df) == 4
    assert df["sentiment_score"].dtype == "float64"
    assert df["distress_flag"].dtype == "int32"
    assert df["complaint_severity"].dtype == "float64"
    assert df["credit_intent_flag"].dtype == "int32"
    assert df.iloc[0]["sentiment_score"] > 0.0
    assert df.iloc[1]["complaint_severity"] >= 0.80
    assert df.iloc[3]["distress_flag"] == 1
