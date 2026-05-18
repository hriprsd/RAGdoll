"""Database connection pool and query helpers."""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DBConfig:
    path: Path
    max_connections: int = 5
    timeout: float = 30.0
    wal_mode: bool = True


class ConnectionPool:
    """Simple SQLite connection pool with WAL mode support."""

    def __init__(self, config: DBConfig):
        self.config = config
        self._connections: list[sqlite3.Connection] = []

    def _create_connection(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.config.path), timeout=self.config.timeout)
        con.row_factory = sqlite3.Row
        if self.config.wal_mode:
            con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    @contextmanager
    def acquire(self):
        """Acquire a connection from the pool."""
        if self._connections:
            con = self._connections.pop()
        else:
            con = self._create_connection()
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            if len(self._connections) < self.config.max_connections:
                self._connections.append(con)
            else:
                con.close()

    def close_all(self) -> None:
        """Close all pooled connections."""
        for con in self._connections:
            con.close()
        self._connections.clear()


def execute_query(pool: ConnectionPool, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    """Execute a query and return results as list of dicts."""
    with pool.acquire() as con:
        rows = con.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def paginate(pool: ConnectionPool, sql: str, offset: int = 0, limit: int = 20) -> dict:
    """Execute a paginated query. Returns {items, offset, limit, has_more}."""
    with pool.acquire() as con:
        # Get total count
        count_sql = f"SELECT COUNT(*) FROM ({sql})"
        total = con.execute(count_sql).fetchone()[0]
        # Get page
        rows = con.execute(f"{sql} LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return {
            "items": [dict(r) for r in rows],
            "offset": offset,
            "limit": limit,
            "total": total,
            "has_more": offset + limit < total,
        }
