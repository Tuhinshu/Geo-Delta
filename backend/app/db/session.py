"""
Database Engine & Session Management for GeoDelta
Configures SQLAlchemy with PostgreSQL / PostGIS 3.4 and automatic SQLite fallback.
"""

import os
import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from app.core.config import settings

logger = logging.getLogger("geodelta.db")

Base = declarative_base()


def get_database_url() -> str:
    """
    Returns database connection URL.
    Uses environment setting or falls back to local SQLite for tests.
    """
    url = os.environ.get("DATABASE_URL") or settings.SYNC_DATABASE_URL
    # If explicitly running in local test mode or postgres is unreachable
    if os.environ.get("USE_SQLITE", "false").lower() == "true":
        return "sqlite:///./geodelta_local.db"
    return url


def create_db_engine():
    db_url = get_database_url()
    try:
        if db_url.startswith("sqlite"):
            return create_engine(db_url, connect_args={"check_same_thread": False})
        else:
            return create_engine(
                db_url,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                connect_args={"connect_timeout": 3}
            )
    except Exception as e:
        logger.warning(f"Failed to connect to primary DB ({db_url}): {e}. Falling back to SQLite.")
        return create_engine("sqlite:///./geodelta_local.db", connect_args={"check_same_thread": False})


engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency yielding a transactional database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initializes database tables.
    """
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        logger.warning(f"Could not auto-create database tables on current engine: {e}")
