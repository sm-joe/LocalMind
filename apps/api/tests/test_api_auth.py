from types import SimpleNamespace

from fastapi.testclient import TestClient

import app.auth as auth_module
from app.main import app


def test_api_routes_require_api_key(monkeypatch):
    monkeypatch.setattr(
        auth_module, "get_settings",
        lambda: SimpleNamespace(localrag_api_key="a" * 40, app_env="production"),
    )
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/chat", json={"question": "hello"}).status_code == 401
    assert client.get("/conversations").status_code == 401
    assert client.get("/conversations/fake/messages").status_code == 401


def test_production_fails_closed_without_key(monkeypatch):
    monkeypatch.setattr(
        auth_module, "get_settings",
        lambda: SimpleNamespace(localrag_api_key="", app_env="production"),
    )
    assert TestClient(app).get("/documents").status_code == 503


def test_explicit_key_accepted_for_valid_request(monkeypatch):
    monkeypatch.setattr(
        auth_module, "get_settings",
        lambda: SimpleNamespace(localrag_api_key="a" * 40, app_env="production"),
    )
    # A valid API key must advance into normal request validation.
    response = TestClient(app).post(
        "/chat",
        json={"question": ""},
        headers={"X-API-Key": "a" * 40},
    )
    assert response.status_code == 422
