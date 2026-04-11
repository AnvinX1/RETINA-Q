"""
RETINA-Q — Database Engine & Session Management (PostgreSQL via SQLAlchemy)

Gracefully degrades when PostgreSQL is not available — the API will still
serve predictions, but scan history and patient records won't be persisted.
"""
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings


# ── Attempt database connection ─────────────────────────────
_db_available = False
engine = None
SessionLocal = None

try:
    engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )
    # Test the connection immediately
    with engine.connect() as conn:
        conn.execute(__import__("sqlalchemy").text("SELECT 1"))
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    _db_available = True
    logger.info("Database connection established")
except Exception as e:
    logger.warning(f"Database unavailable — running without persistence: {e}")


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


def db_available() -> bool:
    """Check whether the database is connected."""
    return _db_available


def get_db():
    """FastAPI dependency — yields a DB session or None if DB is unavailable."""
    if not _db_available or SessionLocal is None:
        yield None
        return
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables if they don't exist."""
    if not _db_available or engine is None:
        logger.warning("Skipping DB table creation — database not available")
        return
    from app.db_models import patient, scan  # noqa: F401 — ensure models are registered
    Base.metadata.create_all(bind=engine)
