from celery import Celery
from config import REDIS_URL

celery_app = Celery(
    "rag_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    imports=("tasks",),  # Import tasks module so Celery can discover tasks
)

# Import tasks to ensure they're registered
import tasks  # noqa: F401

