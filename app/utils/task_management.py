import threading
from datetime import datetime
from typing import Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread-safe storage for scraping progress with locks
scraping_progress: Dict[str, dict] = {}
progress_lock = threading.RLock()  # Re-entrant lock for nested operations

# Cleanup interval for old tasks (in seconds)
CLEANUP_INTERVAL = 3600  # 1 hour
MAX_TASK_AGE = 7200  # 2 hours

def cleanup_old_tasks():
    """Remove old completed or errored tasks from memory."""
    current_time = datetime.now()
    with progress_lock:
        tasks_to_remove = []
        for task_id, task_data in scraping_progress.items():
            task_age = (current_time - task_data["start_time"]).total_seconds()
            if task_age > MAX_TASK_AGE and task_data.get("is_completed", False):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del scraping_progress[task_id]
            logger.info(f"Cleaned up old task: {task_id}")

def get_progress_safely(task_id: str) -> dict:
    """Thread-safe way to get progress data."""
    with progress_lock:
        return scraping_progress.get(task_id, {}).copy()

def update_progress_safely(task_id: str, status: str, **kwargs):
    """Thread-safe progress update function."""
    with progress_lock:
        if task_id in scraping_progress:
            scraping_progress[task_id].update({
                "status": status,
                "last_update": datetime.now(),
                **kwargs
            })
            
            # Set completion flag for final states
            if status in ["completed", "cancelled", "error", "partial_completed"]:
                scraping_progress[task_id]["is_completed"] = True
