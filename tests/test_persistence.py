import pytest

import rpg_bot.persistence.database as database
import rpg_bot.persistence.repository as repo


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "_DB_PATH", tmp_path / "test.db")
    # thread-local connection must be reset so this test gets a fresh one
    database._local.conn = None
    yield database.get_db()


def test_message_ordering_stable_on_timestamp_tie(db):
    chat = repo.create_chat()
    repo.add_message(chat["id"], "user", "first")
    repo.add_message(chat["id"], "user", "second")
    # Force both messages to share a timestamp (second resolution)
    db.execute(
        "UPDATE messages SET created_at = '2020-01-01T00:00:00Z' WHERE chat_id = ?",
        (chat["id"],),
    )
    db.commit()
    data = repo.get_chat(chat["id"])
    assert [m["content"] for m in data["messages"]] == ["first", "second"]


def test_auto_title_from_first_message(db):
    chat = repo.create_chat()
    repo.auto_title(chat["id"], "How do opportunity attacks work?")
    assert repo.get_chat(chat["id"])["title"] == "How do opportunity attacks work?"


def test_auto_title_truncates_at_word_boundary(db):
    chat = repo.create_chat()
    repo.auto_title(chat["id"], "word " * 40)
    title = repo.get_chat(chat["id"])["title"]
    assert title.endswith("...")
    assert title.rstrip(".").split()[-1] == "word"
    assert len(title) <= 53


def test_auto_title_does_not_overwrite_custom_title(db):
    chat = repo.create_chat()
    repo.update_chat(chat["id"], title="Custom")
    repo.auto_title(chat["id"], "something else")
    assert repo.get_chat(chat["id"])["title"] == "Custom"


def test_delete_chat_cascades_messages(db):
    chat = repo.create_chat()
    repo.add_message(chat["id"], "user", "hi")
    assert repo.chat_exists(chat["id"])
    assert repo.delete_chat(chat["id"]) is True
    assert repo.chat_exists(chat["id"]) is False
    assert db.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0


def test_chat_exists(db):
    chat = repo.create_chat()
    assert repo.chat_exists(chat["id"]) is True
    assert repo.chat_exists("nonexistent") is False
