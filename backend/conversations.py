"""
Conversation persistence module.
SQLite-backed storage for chat conversations and messages.
"""
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from backend.config import CONVERSATIONS_DB

logger = logging.getLogger(__name__)


def _get_connection() -> sqlite3.Connection:
    """Get SQLite connection with WAL mode for better concurrency."""
    conn = sqlite3.connect(str(CONVERSATIONS_DB), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Initialize the database schema."""
    conn = _get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New conversation',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                sources TEXT DEFAULT '[]',
                pipeline TEXT DEFAULT '[]',
                time_taken REAL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, created_at);
        """)
        conn.commit()
        logger.info("Conversations database initialized")
    finally:
        conn.close()


# Initialize on module import
init_db()


# ── Conversation CRUD ─────────────────────────────────

def create_conversation(title: str = "New conversation") -> Dict[str, Any]:
    """Create a new conversation."""
    conn = _get_connection()
    try:
        conv_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conv_id, title, now, now),
        )
        conn.commit()
        logger.info(f"Created conversation: {conv_id}")
        return {"id": conv_id, "title": title, "created_at": now, "updated_at": now, "message_count": 0}
    finally:
        conn.close()


def list_conversations() -> List[Dict[str, Any]]:
    """List all conversations ordered by most recently updated."""
    conn = _get_connection()
    try:
        rows = conn.execute("""
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   COUNT(m.id) as message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
        """).fetchall()

        return [
            {
                "id": r["id"],
                "title": r["title"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "message_count": r["message_count"],
            }
            for r in rows
        ]
    finally:
        conn.close()


def get_conversation(conv_id: str) -> Optional[Dict[str, Any]]:
    """Get a conversation with all its messages."""
    conn = _get_connection()
    try:
        conv = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()

        if not conv:
            return None

        messages = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at",
            (conv_id,),
        ).fetchall()

        return {
            "id": conv["id"],
            "title": conv["title"],
            "created_at": conv["created_at"],
            "updated_at": conv["updated_at"],
            "messages": [
                {
                    "id": m["id"],
                    "role": m["role"],
                    "content": m["content"],
                    "sources": json.loads(m["sources"]) if m["sources"] else [],
                    "pipeline": json.loads(m["pipeline"]) if m["pipeline"] else [],
                    "time": m["time_taken"],
                    "created_at": m["created_at"],
                }
                for m in messages
            ],
        }
    finally:
        conn.close()


def update_conversation_title(conv_id: str, title: str) -> bool:
    """Update a conversation's title."""
    conn = _get_connection()
    try:
        now = datetime.now(timezone.utc).isoformat()
        result = conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, conv_id),
        )
        conn.commit()
        return result.rowcount > 0
    finally:
        conn.close()


def delete_conversation(conv_id: str) -> bool:
    """Delete a conversation and all its messages."""
    conn = _get_connection()
    try:
        result = conn.execute(
            "DELETE FROM conversations WHERE id = ?", (conv_id,)
        )
        conn.commit()
        logger.info(f"Deleted conversation: {conv_id}")
        return result.rowcount > 0
    finally:
        conn.close()


# ── Message CRUD ──────────────────────────────────────

def add_message(
    conv_id: str,
    role: str,
    content: str,
    sources: Optional[List[Dict]] = None,
    pipeline: Optional[List[Dict]] = None,
    time_taken: float = 0,
) -> Dict[str, Any]:
    """Add a message to a conversation."""
    conn = _get_connection()
    try:
        msg_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        conn.execute(
            """INSERT INTO messages (id, conversation_id, role, content, sources, pipeline, time_taken, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                msg_id,
                conv_id,
                role,
                content,
                json.dumps(sources or []),
                json.dumps(pipeline or []),
                time_taken,
                now,
            ),
        )

        # Update conversation timestamp
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conv_id),
        )

        # Auto-generate title from first user message
        msg_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ?",
            (conv_id,),
        ).fetchone()["cnt"]

        if msg_count == 1 and role == "user":
            # Use first ~50 chars of first message as title
            title = content[:50].strip()
            if len(content) > 50:
                title += "…"
            conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title, conv_id),
            )

        conn.commit()

        return {
            "id": msg_id,
            "role": role,
            "content": content,
            "sources": sources or [],
            "pipeline": pipeline or [],
            "time": time_taken,
            "created_at": now,
        }
    finally:
        conn.close()
