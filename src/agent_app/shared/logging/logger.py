import logging
import sys
from  pythonjsonlogger import jsonlogger

from shared.logging.formatter import JsonFormatter


_initialized = False


def setup_logging(level: str = "INFO") -> None:
    global _initialized

    if _initialized:
        return

    root = logging.getLogger()
    root.setLevel(level)

    formatter = JsonFormatter()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    root.addHandler(console)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)