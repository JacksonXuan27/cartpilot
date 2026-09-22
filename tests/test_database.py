from pathlib import Path

import pytest

from app.database import (
    DatabaseConfigurationError,
    DatabaseManager,
    DatabaseNotInitializedError,
)


def test_database_manager_requires_open_before_access():
    manager = DatabaseManager("sqlite:///:memory:")

    with pytest.raises(DatabaseNotInitializedError):
        _ = manager.connection


def test_database_manager_reuses_connection_and_commits_transactions(tmp_path: Path):
    database_path = tmp_path / "cartpilot.db"
    manager = DatabaseManager(f"sqlite:///{database_path}")

    first_connection = manager.open()
    assert manager.open() is first_connection
    with manager.transaction() as connection:
        connection.execute("CREATE TABLE events (name TEXT NOT NULL)")
        connection.execute("INSERT INTO events (name) VALUES (?)", ("created",))

    with manager.transaction() as connection:
        row = connection.execute("SELECT name FROM events").fetchone()

    assert row["name"] == "created"
    manager.close()
    assert not manager.is_open


def test_database_manager_rolls_back_failed_transactions():
    manager = DatabaseManager("sqlite:///:memory:")
    manager.open()
    with manager.transaction() as connection:
        connection.execute("CREATE TABLE events (name TEXT NOT NULL)")

    with pytest.raises(RuntimeError, match="write failed"):
        with manager.transaction() as connection:
            connection.execute("INSERT INTO events (name) VALUES (?)", ("failed",))
            raise RuntimeError("write failed")

    with manager.transaction() as connection:
        assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0

    manager.close()


@pytest.mark.asyncio
async def test_database_manager_lifecycle_closes_connection():
    manager = DatabaseManager("sqlite:///:memory:")

    async with manager.lifecycle() as active_manager:
        assert active_manager is manager
        assert manager.is_open

    assert not manager.is_open


def test_database_manager_rejects_unsupported_urls():
    manager = DatabaseManager("mysql://localhost/cartpilot")

    with pytest.raises(DatabaseConfigurationError, match="sqlite"):
        manager.open()
