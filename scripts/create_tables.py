"""
Women Safety AI — Create DB Tables (Non-Docker Fallback)
==========================================================
Uses SQLAlchemy's create_all() to create all tables directly.
Works with both PostgreSQL (via Docker) and SQLite (local dev).

Run::

    python scripts/create_tables.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings


def create_tables():
    """Create all ORM tables in the configured database."""
    from sqlalchemy import create_engine
    from db.models import Base

    engine = create_engine(settings.database_url, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    print(f"✅ All tables created in: {settings.database_url}")
    engine.dispose()


if __name__ == "__main__":
    create_tables()
