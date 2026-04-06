import logging
import sys
from typing import Optional

# Default format used before Config is loaded.
# Config._configure_logging() calls configure_logging() to override this
# from the YAML — avoiding a circular import (config.py imports get_logger).
_DEFAULT_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Module-level state updated by configure_logging()
_active_fmt: str = _DEFAULT_FMT
_active_datefmt: str = _DEFAULT_DATEFMT


def configure_logging(fmt: str, datefmt: str = _DEFAULT_DATEFMT) -> None:
    """
    Update the formatter on all existing 'src.*' logger handlers.

    Called once by Config._configure_logging() after the YAML is parsed,
    so the format is driven by config rather than hardcoded here.
    """
    global _active_fmt, _active_datefmt
    _active_fmt = fmt
    _active_datefmt = datefmt

    new_formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    for name, logger in logging.Logger.manager.loggerDict.items():
        if name.startswith("src") and isinstance(logger, logging.Logger):
            for handler in logger.handlers:
                handler.setFormatter(new_formatter)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Return a structured logger for the given module name.

    Usage:
        logger = get_logger(__name__)
        logger.info("Agent started", extra={"agent": "blog_writer"})
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(fmt=_active_fmt, datefmt=_active_datefmt)
        )
        logger.addHandler(handler)
        logger.propagate = False

    if level is not None:
        logger.setLevel(level)
    elif not logger.level:
        logger.setLevel(logging.INFO)

    return logger
