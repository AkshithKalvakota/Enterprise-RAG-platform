import os
import shutil
import threading
import time

from backend.dependencies import get_vector_db
from utils import config
from utils.logger import setup_logger

logger = setup_logger()


def run_cleanup():
    """
    Deletes documents older than SESSION_TTL_HOURS: their chunks in Qdrant and their
    upload folders on disk. Covers users who closed the browser without "Remove My Documents".
    """
    try:
        deleted = get_vector_db().delete_expired(config.SESSION_TTL_HOURS)
        if deleted:
            logger.info(f"Cleanup: deleted ~{deleted} chunks older than {config.SESSION_TTL_HOURS:g}h.")
    except Exception as e:
        logger.error(f"Cleanup of old chunks failed: {e}")

    cutoff = time.time() - config.SESSION_TTL_HOURS * 3600
    if os.path.isdir(config.UPLOAD_DIR):
        for name in os.listdir(config.UPLOAD_DIR):
            folder = os.path.join(config.UPLOAD_DIR, name)
            if os.path.isdir(folder) and os.path.getmtime(folder) < cutoff:
                shutil.rmtree(folder, ignore_errors=True)


def _cleanup_loop():
    while True:
        run_cleanup()
        time.sleep(config.CLEANUP_INTERVAL_MINUTES * 60)


def start_cleanup_thread():
    """
    Runs the cleanup now and then every CLEANUP_INTERVAL_MINUTES in the background.
    On Render's free tier the server sleeps when idle, so it also runs on every wake-up
    (including the twice-daily keep-alive call).
    """
    threading.Thread(target=_cleanup_loop, name="session-cleanup", daemon=True).start()
