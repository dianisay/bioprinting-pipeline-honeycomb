"""Centralized logging configuration for the bioprinting pipeline.

Usage in any module:
    from utils.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Message here")

Logs are written to both console and a rotating file (logs/pipeline.log).
"""

import logging
import logging.handlers
import sys
from pathlib import Path


LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "pipeline.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file
BACKUP_COUNT = 3


def _setup_root_logger():
    """Configure root logger once (idempotent)."""
    root = logging.getLogger("bioprint")
    if root.handlers:
        return root

    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console handler (INFO+)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler (DEBUG+, rotating)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    return root


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the 'bioprint' namespace."""
    _setup_root_logger()
    return logging.getLogger(f"bioprint.{name}")
