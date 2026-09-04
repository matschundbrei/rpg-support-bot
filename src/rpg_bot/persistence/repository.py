"""CRUD operations for chats and messages."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from rpg_bot.persistence.database import get_db


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _row_to_dict(row) -> dict:
    return dict(row) if row else {}


def list_chats() -> list[dict]:
    db = get_db()
    rows = db.execute(
        "SELECT id, title, game_system, created_at, updated_at "
        "FROM chats ORDER BY updated_at DESC"
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def create_chat(game_system: str | None = None) -> dict:
    db = get_db()
    chat_id = uuid.uuid4().hex
    now = _now()
    db.execute(
        "INSERT INTO chats (id, title, game_system, created_at, updated_at) "
        "VALUES (?, 'New Chat', ?, ?, ?)",
        (chat_id, game_system, now, now),
    )
    db.commit()
    return {"id": chat_id, "title": "New Chat", "game_system": game_system,
            "created_at": now, "updated_at": now}


def chat_exists(chat_id: str) -> bool:
    db = get_db()
    row = db.execute("SELECT 1 FROM chats WHERE id = ?", (chat_id,)).fetchone()
    return row is not None


def get_chat(chat_id: str) -> dict | None:
    db = get_db()
    chat = db.execute(
        "SELECT id, title, game_system, created_at, updated_at "
        "FROM chats WHERE id = ?",
        (chat_id,),
    ).fetchone()
    if not chat:
        return None

    messages = db.execute(
        "SELECT id, role, content, created_at "
        "FROM messages WHERE chat_id = ? ORDER BY created_at, rowid",
        (chat_id,),
    ).fetchall()

    result = _row_to_dict(chat)
    result["messages"] = [_row_to_dict(m) for m in messages]
    return result


def update_chat(
    chat_id: str,
    title: str | None = None,
    game_system: str | None = None,
) -> dict | None:
    db = get_db()
    chat = db.execute("SELECT id FROM chats WHERE id = ?", (chat_id,)).fetchone()
    if not chat:
        return None

    updates = []
    params = []
    if title is not None:
        updates.append("title = ?")
        params.append(title)
    if game_system is not None:
        updates.append("game_system = ?")
        params.append(game_system)
    if not updates:
        return get_chat(chat_id)

    updates.append("updated_at = ?")
    params.append(_now())
    params.append(chat_id)

    db.execute(f"UPDATE chats SET {', '.join(updates)} WHERE id = ?", params)
    db.commit()
    return get_chat(chat_id)


def delete_chat(chat_id: str) -> bool:
    db = get_db()
    cursor = db.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    db.commit()
    return cursor.rowcount > 0


def add_message(chat_id: str, role: str, content: str) -> dict:
    db = get_db()
    msg_id = uuid.uuid4().hex
    now = _now()
    db.execute(
        "INSERT INTO messages (id, chat_id, role, content, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (msg_id, chat_id, role, content, now),
    )
    db.execute(
        "UPDATE chats SET updated_at = ? WHERE id = ?",
        (now, chat_id),
    )
    db.commit()
    return {"id": msg_id, "role": role, "content": content, "created_at": now}


def auto_title(chat_id: str, first_user_message: str) -> None:
    """Set chat title from first user message if still 'New Chat'."""
    db = get_db()
    chat = db.execute(
        "SELECT title FROM chats WHERE id = ?", (chat_id,)
    ).fetchone()
    if not chat or chat["title"] != "New Chat":
        return

    title = first_user_message.strip()[:50]
    # Truncate at last word boundary
    if len(first_user_message.strip()) > 50 and " " in title:
        title = title[: title.rfind(" ")] + "..."
    elif len(first_user_message.strip()) > 50:
        title += "..."

    db.execute(
        "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
        (title, _now(), chat_id),
    )
    db.commit()
