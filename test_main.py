import os
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

# Mock token for testing (replace with valid token for integration)
MOCK_TOKEN = "test-token"
HEADERS = {"Authorization": f"Bearer {MOCK_TOKEN}"}

def test_memory_write_and_read(monkeypatch):
    # Patch Google token verification to always pass
    monkeypatch.setattr("main.verify_google_token", lambda: {"email": "test@infinity.com"})
    data = {
        "agent_id": "test-agent",
        "scope": "test-scope",
        "importance": 5,
        "confidence": 0.9,
        "content": {"msg": "hello"},
        "tags": ["unit"],
        "source": "pytest"
    }
    write_resp = client.post("/memory/write", json=data, headers=HEADERS)
    assert write_resp.status_code == 200
    mem_id = write_resp.json()["id"]
    # Search
    search_resp = client.post("/memory/search", json={"query": "hello", "scope": "test-scope"}, headers=HEADERS)
    assert search_resp.status_code == 200
    assert any("hello" in str(r["content"]) for r in search_resp.json()["results"])
