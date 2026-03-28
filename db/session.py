"""
Women Safety AI — Database Session Factory
============================================
Provides the SQLAlchemy engine, session factory, and a FastAPI
dependency generator ``get_db()`` that yields a session per request.

Falls back to SQLite if PostgreSQL is not available.
"""

import os

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from config.settings import settings

# ── Determine database URL ──
_db_url = settings.database_url

# If PostgreSQL is configured but not reachable, fall back to SQLite
try:
    _engine = create_engine(_db_url, pool_pre_ping=True)
    with _engine.connect() as conn:
        conn.execute(_engine.dialect.server_version_info if hasattr(_engine.dialect, 'server_version_info') else conn.execute)
    engine = _engine
except Exception:
    # Fall back to SQLite for local dev without Docker
    _sqlite_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "women_safety.db",
    )
    os.makedirs(os.path.dirname(_sqlite_path), exist_ok=True)
    _db_url = f"sqlite:///{_sqlite_path}"
    engine = create_engine(_db_url, connect_args={"check_same_thread": False})

    # Enable foreign keys for SQLite
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    # Auto-create tables with SQLite
    from db.models import Base
    Base.metadata.create_all(engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI dependency — yields a DB session, closes on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
