import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger as _logger  # pylint: disable=import-error

PRINT_LEVEL = "INFO"


def define_log_level(
    print_level: str = "INFO", logfile_level: str = "DEBUG", name: Optional[str] = None
) -> Any:
    """Adjust the log level to above level."""
    global PRINT_LEVEL  # pylint: disable=global-statement
    PRINT_LEVEL = print_level

    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d%H%M%S")
    log_name = f"{name}_{formatted_date}" if name else formatted_date  # name a log with prefix name

    # Get workdir from environment or default to avoid circular imports
    workdir = os.getenv("AGRAPH_WORKDIR", "workdir")

    _logger.remove()
    _logger.add(sys.stderr, level=print_level)
    _logger.add(Path(workdir) / f"logs/{log_name}.log", level=logfile_level)
    return _logger


logger = define_log_level()


if __name__ == "__main__":
    logger.info("Starting application")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    try:
        raise ValueError("Test error")
    except Exception as e:  # pylint:disable=broad-except
        logger.exception(f"An error occurred: {e}")
