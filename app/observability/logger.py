import logging
import json
import sys
from datetime import datetime


# Reserved LogRecord attributes that cannot be overwritten
_RESERVED_ATTRS = {
    "name", "msg", "args", "levelname", "levelno",
    "pathname", "filename", "module", "exc_info",
    "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread",
    "threadName", "processName", "process", "message"
}


class JSONFormatter(logging.Formatter):
    """
    Production-safe JSON formatter.

    Guarantees:
    • Never crashes
    • Preserves structured logging
    • Allows custom extra fields safely
    """

    def format(self, record: logging.LogRecord) -> str:

        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Safely add custom extra fields
        for key, value in record.__dict__.items():

            if key.startswith("_"):
                continue

            if key in _RESERVED_ATTRS:
                continue

            # Avoid overwriting existing fields
            if key in log_data:
                log_data[f"extra_{key}"] = value
            else:
                log_data[key] = value

        # Exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logging(log_level: str = "INFO"):

    import os
    os.makedirs("logs", exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    root_logger.handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)

    # Silence noisy libs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_request_start(logger, request_id, endpoint, **kwargs):

    safe_extra = {
        "request_id": request_id,
        "endpoint": endpoint,
        **kwargs
    }

    logger.info(f"{endpoint}_started", extra=safe_extra)


def log_request_complete(logger, request_id, endpoint, latency_seconds, **kwargs):

    safe_extra = {
        "request_id": request_id,
        "endpoint": endpoint,
        "latency_seconds": round(latency_seconds, 3),
        **kwargs
    }

    logger.info(f"{endpoint}_completed", extra=safe_extra)


def log_request_error(logger, request_id, endpoint, error, **kwargs):

    safe_extra = {
        "request_id": request_id,
        "endpoint": endpoint,
        "error": str(error),
        "error_type": type(error).__name__,
        **kwargs
    }

    logger.error(
        f"{endpoint}_failed",
        extra=safe_extra,
        exc_info=True
    )
