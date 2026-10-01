"""SQLite connection handling.

A single connection per thread; the worker pool runs each company on its own
thread so research state stays isolated while the shared cache remains usable.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

from leadfinder.storage.schema import SCHEMA


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute("PRAGMA foreign_keys = ON")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # -- write helpers ---------------------------------------------------
    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        conn = self.connect()
        with self._write_lock:
            cursor = conn.execute(sql, tuple(params))
            conn.commit()
            return cursor

    def executemany(self, sql: str, seq: Iterable[Iterable[Any]]) -> None:
        conn = self.connect()
        with self._write_lock:
            conn.executemany(sql, [tuple(item) for item in seq])
            conn.commit()

    def insert(self, sql: str, params: Iterable[Any] = ()) -> int:
        cursor = self.execute(sql, params)
        # An INSERT OR IGNORE that hits a unique constraint reports rowcount 0.
        # lastrowid is connection-wide, so without this guard it would leak the
        # previous successful insert's id and attach evidence to a person row
        # that does not exist.
        if cursor.rowcount == 0:
            return 0
        return int(cursor.lastrowid or 0)

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return list(self.connect().execute(sql, tuple(params)).fetchall())

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        return self.connect().execute(sql, tuple(params)).fetchone()

    def scalar(self, sql: str, params: Iterable[Any] = ()) -> Any:
        row = self.query_one(sql, params)
        return row[0] if row else None
