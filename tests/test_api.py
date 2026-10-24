"""Tests for FastAPI endpoints, request validation, error responses, and health checks."""

import pytest
from fastapi.testclient import TestClient


def test_root_endpoint(test_client: TestClient):
    response = test_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "TicketLens IT Ticket Classification API"
    assert data["status"] == "healthy"


def test_health_endpoint(test_client: TestClient):
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "ready" in data
    assert "models_loaded" in data


def test_metadata_endpoint(test_client: TestClient):
    response = test_client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["project_name"] == "TicketLens"
    assert "target_categories" in data
    assert "Incident" in data["target_categories"]


def test_predict_validation_error_empty_text(test_client: TestClient):
    # Empty string should fail validation
    response = test_client.post("/predict", json={"text": "   "})
    assert response.status_code == 422


def test_predict_validation_error_too_short(test_client: TestClient):
    response = test_client.post("/predict", json={"text": "hi"})
    assert response.status_code == 422


def test_predict_fallback_when_models_not_loaded(monkeypatch, test_client: TestClient):
    import ticketlens.api as api_module
    # Simulate uninitialized predictor
    monkeypatch.setattr(api_module, "predictor", None)
    response = test_client.post("/predict", json={"text": "VPN connection broken"})
    assert response.status_code == 503
    assert "No trained model artifacts found" in response.json()["detail"]


def test_predict_success_transformer(test_client: TestClient):
    response = test_client.post(
        "/predict",
        json={"text": "VPN tunnel is unexpectedly disconnected and cannot reconnect."},
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_category" in data
    assert "confidence" in data
    assert "decision" in data
    assert data["decision"] in ["Auto Accept", "Needs Review"]
    assert "probabilities" in data
    assert len(data["probabilities"]) == 4


def test_predict_success_baseline(test_client: TestClient):
    response = test_client.post(
        "/predict",
        json={
            "text": "Please provide read-only database credentials for the new analyst.",
            "model_type": "baseline",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "TF-IDF + Logistic Regression"
    assert "predicted_category" in data
    assert data["decision"] in ["Auto Accept", "Needs Review"]

