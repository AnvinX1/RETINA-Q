"""
RETINA-Q — Celery Application

Configures an async task queue backed by Redis for offloading
heavy ML / quantum inference from the HTTP request cycle.
"""
from celery import Celery
from app.config import settings

celery_app = Celery(
    "retinaq",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks"],
)

# ── Celery configuration ────────────────────────────────────
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    result_expires=3600,            # Results expire after 1 hour
    task_track_started=True,        # Allow "STARTED" state
    worker_prefetch_multiplier=1,   # One task at a time per worker (GPU bound)
    task_acks_late=True,            # Ack after completion (crash safety)
    timezone="UTC",
)
