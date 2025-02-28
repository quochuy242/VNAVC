import sys

from loguru import logger

logger.add(
    sys.stdout,
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    level="DEBUG",
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
)
