import logging
from logging.handlers import RotatingFileHandler
import os

# Ensure log directory exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file path
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Set up rotating log handler (max 5 MB per file, keep 3 backups)
handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3)

# Log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# Main logger setup
logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)  # Or DEBUG
logger.addHandler(handler)
logger.propagate = False  # Avoid double logging
