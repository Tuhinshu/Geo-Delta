"""
Celery Task Queue Configuration for GeoDelta GEOINT Platform
Manages asynchronous GPU inference, COG streaming, and reporting jobs.
"""

from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "geodelta_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.workers.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,        # Hard timeout 5 minutes
    task_soft_time_limit=240,   # Soft timeout 4 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,  # Flush worker process periodically to prevent GPU memory leaks
    result_expires=86400        # Keep task results for 24 hours
)

if __name__ == "__main__":
    celery_app.start()
