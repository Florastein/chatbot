"""Local assistant services. No third-party Python dependencies.

Public API re-exported here for backward compatibility:
  Store, CustomAssistant, find_files, read_attachment, open_app
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from custom_assistant import CustomAssistant
from tool_actions import open_application as open_app

BASE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# SQLite store — messages, notes/tasks, settings
# ---------------------------------------------------------------------------

class Store:
    def __init__(self, path=BASE / "data" / "assistant.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, role TEXT, content TEXT);
                CREATE TABLE IF NOT EXISTS items(id INTEGER PRIMARY KEY, kind TEXT, content TEXT, done INTEGER DEFAULT 0);
                CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path)
        try:
            with db:
                yield db
        finally:
            db.close()

    def messages(self):
        with self.connect() as db:
            return [dict(role=r, content=c) for r, c in db.execute("SELECT role,content FROM messages ORDER BY id")]

    def append(self, role, content):
        with self.connect() as db:
            db.execute("INSERT INTO messages(role,content) VALUES (?,?)", (role, content))

    def clear_chat(self):
        with self.connect() as db:
            db.execute("DELETE FROM messages")

    def items(self, kind):
        with self.connect() as db:
            return db.execute("SELECT id,content,done FROM items WHERE kind=? ORDER BY id DESC", (kind,)).fetchall()

    def add(self, kind, content):
        if not content.strip():
            raise ValueError("Enter some text first.")
        with self.connect() as db:
            db.execute("INSERT INTO items(kind,content) VALUES (?,?)", (kind, content.strip()))

    def toggle(self, item_id):
        with self.connect() as db:
            db.execute("UPDATE items SET done=1-done WHERE id=?", (item_id,))

    def delete(self, item_id):
        with self.connect() as db:
            db.execute("DELETE FROM items WHERE id=?", (item_id,))

    def update(self, item_id, content):
        if not content.strip():
            raise ValueError("Enter some text first.")
        with self.connect() as db:
            db.execute("UPDATE items SET content=? WHERE id=?", (content.strip(), item_id))

    def setting(self, key, default=""):
        with self.connect() as db:
            row = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_setting(self, key, value):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, value))


# ---------------------------------------------------------------------------
# File helpers (cross-platform)
# ---------------------------------------------------------------------------

def find_files(folder=None, query="", limit=200):
    """Search a folder, defaulting to the active user's home directory."""
    folder = folder or Path.home()
    if not Path(folder).is_dir() or not query.strip():
        raise ValueError("Choose a folder and enter part of a filename or folder name.")
    from tool_actions import search_files
    result = search_files(folder, query, limit)
    return result["paths"], result["limited"]


def read_attachment(path):
    """Read a text file and wrap it for the assistant."""
    path = Path(path)
    if path.stat().st_size > 24000:
        raise ValueError("Select a text file smaller than 24 KB, or paste a relevant excerpt.")
    text = path.read_text(encoding="utf-8")
    if "\x00" in text:
        raise ValueError("Select a plain-text or source-code file.")
    return f"Attached file: {path.name}\n<file_content>\n{text}\n</file_content>"
