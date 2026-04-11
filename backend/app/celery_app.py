"""
RETINA-Q — Celery Application

Configures an async task queue backed by Redis for offloading
heavy ML / quantum inference from the HTTP request cycle.

Gracefully degrades when Redis is not available — celery_app will be None
and the predict routes will fall back to synchronous inference.
"""
from loguru import logger

celery_app = None

try:
    from celery import Celery
    from app.config import settings

    _app = Celery(
        "retinaq",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
        include=["app.tasks"],
    )

    # ── Celery configuration ────────────────────────────────────
    _app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        result_expires=3600,            # Results expire after 1 hour
        task_track_started=True,        # Allow "STARTED" state
        worker_prefetch_multiplier=1,   # One task at a time per worker (GPU bound)
        task_acks_late=True,            # Ack after completion (crash safety)
        timezone="UTC",
        broker_connection_retry_on_startup=False,
        broker_connection_timeout=3,    # Fail fast if Redis isn't there
    )

    # Test connectivity by attempting a brief ping
    try:
        conn = _app.connection()
        conn.ensure_connection(max_retries=1, timeout=2)
        conn.close()
        celery_app = _app
        logger.info("Celery/Redis connection established")
    except Exception as e:
        logger.warning(f"Redis/Celery unavailable — async mode disabled: {e}")
        celery_app = None

except Exception as e:
    logger.warning(f"Celery setup failed — async mode disabled: {e}")
    celery_app = None
