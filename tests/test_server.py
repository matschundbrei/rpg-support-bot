import types

import pytest
from fastapi.testclient import TestClient

import rpg_bot.api.server as server
import rpg_bot.persistence.database as database


def _fake_settings(api_key: str):
    return types.SimpleNamespace(api_key=api_key)


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Point the repository at a temp DB so tests don't touch data/chats.db
    monkeypatch.setattr(database, "_DB_PATH", tmp_path / "test.db")
    database._local.conn = None
    return TestClient(server.app)


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.setattr(server, "get_settings", lambda: _fake_settings(""))


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setattr(server, "get_settings", lambda: _fake_settings("secret"))


def test_api_open_when_no_key_configured(client, no_key):
    assert client.get("/api/chats").status_code == 200
    created = client.post("/api/chats", json={})
    assert created.status_code == 201
    assert client.get("/v1/models").status_code == 200


def test_api_requires_key_when_configured(client, with_key):
    assert client.get("/api/chats").status_code == 401
    assert client.get("/v1/models").status_code == 401
    assert client.post("/api/chats", json={}).status_code == 401


def test_api_accepts_valid_bearer_key(client, with_key):
    headers = {"Authorization": "Bearer secret"}
    assert client.get("/api/chats", headers=headers).status_code == 200
    created = client.post("/api/chats", json={}, headers=headers)
    assert created.status_code == 201


def test_api_rejects_wrong_key(client, with_key):
    headers = {"Authorization": "Bearer wrong"}
    assert client.get("/api/chats", headers=headers).status_code == 401


def test_ui_paths_stay_public(client, with_key):
    # The web UI shell must load without a key so the browser can prompt
    assert client.get("/").status_code == 200
