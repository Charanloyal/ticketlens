"""Tests for FastAPI endpoints, request validation, error responses, and health checks."""

import pytest
from fastapi.testclient import TestClient


def test_root_endpoint_html(test_client: TestClient):
    response = test_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "TicketLens" in response.text
    assert "Ticket Ingestion Workbench" in response.text


def test_root_endpoint_json(test_client: TestClient):
    response = test_client.get("/?format=json")
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


def test_head_requests_supported_for_uptime_monitors(test_client: TestClient):
    # Render's wake-up page and uptime monitors (e.g. UptimeRobot) probe with HEAD.
    for path in ("/", "/demo", "/health"):
        response = test_client.head(path)
        assert response.status_code == 200, path
        assert response.content == b""


def test_health_returns_503_when_no_model(monkeypatch, test_client: TestClient):
    import ticketlens.api as api_module
    monkeypatch.setattr(api_module, "predictor", None)
    response = test_client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "ready": False, "models_loaded": []}


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
    assert isinstance(data["itil_cues"], dict)
    assert data["warnings"] == []


def test_predict_handwritten_invoice_is_request(test_client: TestClient):
    response = test_client.post("/predict", json={"text": "send me last month's invoice", "model_type": "baseline"})
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_category"] == "Request"
    assert "Request" in data["itil_cues"]


def test_predict_returns_routing_and_redacted_text(test_client: TestClient):
    data = test_client.post(
        "/predict",
        json={"text": "Hi, I'm john.doe@acme.com, call 555-123-4567. Server 10.0.0.12 is down right now.", "model_type": "baseline"},
    ).json()
    assert "john.doe@acme.com" not in data["processed_text"]
    for token in ("[EMAIL]", "[PHONE]", "[IP]"):
        assert token in data["processed_text"]
    route = data["routing"]
    assert route["auto_routed"] == (data["decision"] == "Auto Accept")
    if route["auto_routed"]:
        assert route["queue"] == route["suggested_queue"]
    else:
        assert route["queue"] == "Human Triage Queue"
    assert route["response_target"]


def test_auto_accepted_ticket_is_routed_to_its_category_queue(test_client: TestClient):
    data = test_client.post(
        "/predict",
        json={"text": "Please upgrade all office laptops from Windows 10 to Windows 11 next weekend.", "model_type": "baseline"},
    ).json()
    assert data["decision"] == "Auto Accept"
    assert data["routing"] == {
        "queue": "Change Advisory Board (CAB)",
        "response_target": "Assessed at the next CAB meeting",
        "suggested_queue": "Change Advisory Board (CAB)",
        "auto_routed": True,
    }


def test_redaction_can_be_disabled(test_client: TestClient):
    data = test_client.post(
        "/predict", json={"text": "Mail bob@corp.com about the VPN outage", "redact_pii": False, "model_type": "baseline"}
    ).json()
    assert "bob@corp.com" in data["processed_text"]


def test_predict_non_english_forces_review(test_client: TestClient):
    response = test_client.post(
        "/predict",
        json={"text": "Der Drucker im Büro funktioniert nicht und ich kann nicht drucken"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "Needs Review"
    assert any("German" in w for w in data["warnings"])


def test_api_key_not_required_by_default(monkeypatch, test_client: TestClient):
    monkeypatch.delenv("TICKETLENS_API_KEY", raising=False)
    assert test_client.get("/metadata").json()["auth_required"] is False
    response = test_client.post("/predict", json={"text": "VPN connection broken since morning"})
    assert response.status_code == 200


def test_api_key_enforced_when_configured(monkeypatch, test_client: TestClient):
    monkeypatch.setenv("TICKETLENS_API_KEY", "s3cret-key")
    body = {"text": "VPN connection broken since morning"}

    assert test_client.post("/predict", json=body).status_code == 401
    assert test_client.post("/predict", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    assert test_client.post("/predict", json=body, headers={"X-API-Key": "s3cret-key"}).status_code == 200

    # Health checks and metadata stay public so Render's health check keeps working.
    assert test_client.get("/health").status_code == 200
    assert test_client.get("/metadata").json()["auth_required"] is True

