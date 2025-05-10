from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "worker", # Set a default name for the worker
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=['app.tasks.summarization_tasks'] # List of modules to import when worker starts
)

# Optional Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],  # Ignore other content
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    # You might want to set other configurations like task_track_started=True
    task_track_started=True, # Useful for updating status to PROCESSING
    # Consider result_expires for cleaning up results from Redis
    result_expires=3600, # Expire results after 1 hour (in seconds)
)

if __name__ == '__main__':
    # This is for running the worker directly, e.g. `python -m app.core.celery_app worker -l info`
    # However, it's more common to use `celery -A app.core.celery_app worker -l info`
    celery_app.start() 