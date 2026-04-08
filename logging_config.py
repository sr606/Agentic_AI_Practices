import sys
from loguru import logger
from pathlib import Path


def setup_logger() -> None:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger.remove()

    # Console logger
    logger.add(
        sys.stdout,
        level="INFO",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | "
            "{name}:{function}:{line} | {message}"
        ),
    )

    # File logger
    logger.add(
        log_dir / "pipeline.log",
        level="DEBUG",
        rotation="10 MB",
        retention="14 days",
        enqueue=True,
        backtrace=True,
        diagnose=True,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | "
            "{name}:{function}:{line} | {message}"
        ),
    )