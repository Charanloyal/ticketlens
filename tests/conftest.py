"""Pytest fixtures and configuration for TicketLens."""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from ticketlens.api import app
from ticketlens.config import get_project_root, load_config


@pytest.fixture(scope="session")
def project_root():
    return get_project_root()


@pytest.fixture(scope="session")
def base_config():
    return load_config()


@pytest.fixture(scope="session")
def sample_ticket_fixture(project_root):
    fixture_path = project_root / "data" / "fixtures" / "synthetic_tickets.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def test_client():
    with TestClient(app) as client:
        yield client
