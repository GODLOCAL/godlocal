"""tests/test_api.py — FastAPI endpoint integration tests (godlocal_v6.py)"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_brain():
    brain = MagicMock()
    brain.think = AsyncMock(return_value="mocked LLM response")
    brain.memory = MagicMock()
    brain.memory.short = MagicMock()
    brain.memory.short.count = MagicMock(return_value=5)
    brain.memory.long = MagicMock()
    brain.memory.long.count = MagicMock(return_value=10)
    brain.memory.add = MagicMock()
    brain.memory.prune = MagicMock(return_value=0)
    return brain


@pytest.fixture
def client(mock_brain, tmp_path):
    """TestClient with mocked Brain and scheduler."""
    soul = tmp_path / "BOH_OASIS.md"
    soul.write_text("Test soul", encoding="utf-8")

    with patch("core.brain.Brain.get", return_value=mock_brain),          patch("sleep_scheduler_v6.start_scheduler", return_value=MagicMock()),          patch("core.brain.settings") as ms:
        ms.model = "qwen3:8b"
        ms.soul_file = str(soul)
        ms.api_key = ""
        ms.autogenesis_apply = False

        from godlocal_v6 import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestStatusEndpoint:
    def test_status_200(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
        assert data["version"] == "6.0.0"
        assert "model" in data

    def test_status_has_fep(self, client):
        resp = client.get("/status")
        data = resp.json()
        assert "fep" in data


class TestThinkEndpoint:
    def test_think_returns_response(self, client, mock_brain):
        resp = client.post("/think", json={"task": "What is 2+2?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert data["response"] == "mocked LLM response"

    def test_think_empty_task_rejected(self, client):
        resp = client.post("/think", json={"task": ""})
        assert resp.status_code == 422  # Pydantic validation

    def test_think_requires_task(self, client):
        resp = client.post("/think", json={})
        assert resp.status_code == 422


class TestAgentEndpoints:
    def test_agent_status_200(self, client):
        resp = client.get("/agent/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "slots" in data
        assert "active" in data

    def test_agent_swap_unknown_returns_400(self, client):
        resp = client.post("/agent/swap/nonexistent_agent")
        assert resp.status_code == 400


class TestMobileEndpoints:
    def test_mobile_status_200(self, client):
        resp = client.get("/mobile/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "model" in data
        assert "fep" in data


class TestFeedbackEndpoint:
    def test_feedback_records_correction(self, client):
        resp = client.post("/feedback?was_corrected=true")
        assert resp.status_code == 200
        data = resp.json()
        assert "fep" in data


class TestMemoryEndpoints:
    def test_memory_add(self, client):
        resp = client.post("/memory/add", json={"text": "test memory item", "long": False})
        assert resp.status_code == 200

    def test_memory_clear(self, client):
        resp = client.post("/memory/clear")
        assert resp.status_code == 200


class TestAuthMiddleware:
    def test_no_auth_required_when_key_empty(self, client):
        """With empty api_key, no auth header needed."""
        resp = client.post("/think", json={"task": "hello"})
        assert resp.status_code == 200
