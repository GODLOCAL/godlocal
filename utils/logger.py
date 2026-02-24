"""utils/logger.py — structured logging for БОГ || OASIS v6
Supports: colored console, optional JSON (log_json=true in .env)
Usage: from utils.logger import setup_logging; setup_logging()
"""
import logging
import logging.config
import sys
from core.settings import settings

COLORS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
    "RESET":    "\033[0m",
}


class ColorFormatter(logging.Formatter):
    FMT = "%(asctime)s %(levelname)-8s %(name)s  %(message)s"
    DATE = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        color = COLORS.get(record.levelname, "")
        reset = COLORS["RESET"]
        fmt = f"{color}{self.FMT}{reset}"
        formatter = logging.Formatter(fmt, datefmt=self.DATE)
        return formatter.format(record)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        import json, datetime
        return json.dumps({
            "ts":    datetime.datetime.utcnow().isoformat(),
            "level": record.levelname,
            "name":  record.name,
            "msg":   record.getMessage(),
        })


def setup_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if settings.log_json else ColorFormatter())
    logging.root.handlers = []
    logging.root.addHandler(handler)
    logging.root.setLevel(level)
    # Quiet noisy libs
    for noisy in ("chromadb", "httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
