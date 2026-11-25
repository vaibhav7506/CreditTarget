"""Tests for FastAPI serving layer: /v1/score, validation errors, /health, /metrics."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from credittarget.serving.api import app


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient fixture with isolated mock models."""
    from credittarget.serving.api import app_state

    class MockModel:
        def predict_proba(self, X: Any) -> Any:
            import numpy as np

            return np.array([[0.75, 0.25]])

    with TestClient(app) as test_client:
        app_state["prop_model"] = MockModel()
        app_state["pd_model"] = MockModel()
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    """Verify /health returns 200 OK and status HEALTHY."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "propensity_model_version" in data


def test_metrics_endpoint(client: TestClient) -> None:
    """Verify /metrics returns Prometheus metric exposition format."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "credittarget_score_requests_total" in response.text


def test_score_customer_endpoint(client: TestClient) -> None:
    """Verify /v1/score returns valid EV, decisions, and scores for a customer."""
    payload = {
        "customer_id": "CUST_9999999",
        "region_code": "NORTH",
        "age_band": "26-35",
        "income_tier": "HIGH",
        "product_segment": "PREMIER",
        "tenure_months": 48,
        "credit_limit": 15000.0,
        "spend_30d": 1200.0,
        "spend_60d": 2500.0,
        "spend_90d": 3800.0,
        "txn_count_30d": 12,
        "txn_count_90d": 35,
        "spend_std_90d": 45.0,
        "spend_velocity_ratio": 0.95,
        "mean_util_30d": 0.25,
        "mean_util_90d": 0.22,
        "max_util_90d": 0.45,
        "util_trend_30d_vs_90d": 0.03,
        "max_dpd_90d": 0,
        "dpd_events_90d": 0,
        "dpd_events_30d": 0,
        "severe_dpd_flag_90d": 0,
        "note_count_90d": 1,
        "mean_sentiment_90d": 0.40,
        "max_complaint_severity_90d": 0.0,
        "distress_flag_30d": 0,
        "credit_intent_flag_30d": 1,
        "prior_campaign_contacts": 1,
        "prior_acceptance_rate": 0.0,
        "base_interest_yield": 0.10,
        "expected_fee_revenue": 150.0,
        "ead_multiplier": 0.60,
        "promo_cost": 25.0,
    }

    response = client.post("/v1/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "CUST_9999999"
    assert "propensity_score" in data
    assert "pd_score" in data
    assert "expected_net_value" in data
    assert data["target_decision"] in ("TARGET", "SUPPRESS")
    assert "suppression_reason" in data


def test_score_customer_validation_error(client: TestClient) -> None:
    """Verify missing required fields triggers 422 Unprocessable Entity."""
    invalid_payload = {"customer_id": "CUST_1"}
    response = client.post("/v1/score", json=invalid_payload)
    assert response.status_code == 422
