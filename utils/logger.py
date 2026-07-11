from __future__ import annotations

import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from pathlib import Path
from utils.config import settings


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "event"):
            payload["event"] = record.event
        if hasattr(record, "meta"):
            payload["meta"] = record.meta
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
        
    logger.setLevel(settings.log_level.upper())
    
    # 1. Rotating File Handler (Structured JSON Logs)
    log_dir = Path(settings.logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB limit
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(JsonFormatter())
    file_handler.setLevel(settings.log_level.upper())
    logger.addHandler(file_handler)
    
    # 2. Console Handler (Readable human text logs to stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(settings.log_level.upper())
    logger.addHandler(console_handler)
    
    logger.propagate = False
    return logger
