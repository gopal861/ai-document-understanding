# app/observability/logger.py
import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict

class JSONFormatter(logging.Formatter):
    """
    Format logs as JSON for structured logging.
    
    Each log entry is a single JSON object that can be:
    - Parsed by log analysis tools
    - Filtered and searched efficiently
    - Aggregated for metrics
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        
        if hasattr(record, "document_id"):
            log_data["document_id"] = record.document_id
        
        if hasattr(record, "latency_seconds"):
            log_data["latency_seconds"] = record.latency_seconds
        
        if hasattr(record, "confidence_score"):
            log_data["confidence_score"] = record.confidence_score
        
        if hasattr(record, "refused"):
            log_data["refused"] = record.refused
        
        if hasattr(record, "error_type"):
            log_data["error_type"] = record.error_type
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


def setup_logging(log_level: str = "INFO"):
    """
    Configure structured logging for the application.
    
    Creates both console and file handlers with JSON formatting.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    
    # Create logs directory if it doesn't exist
    import os
    os.makedirs("logs", exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(console_handler)
    
    # File handler (persistent logs)
    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_request_start(logger: logging.Logger, request_id: str, endpoint: str, **kwargs):
    """
    Log the start of an API request.
    
    Args:
        logger: Logger instance
        request_id: Unique request identifier
        endpoint: API endpoint being called
        **kwargs: Additional context (document_id, question_length, etc.)
    """
    logger.info(
        f"{endpoint}_started",
        extra={
            "request_id": request_id,
            "endpoint": endpoint,
            **kwargs
        }
    )


def log_request_complete(
    logger: logging.Logger,
    request_id: str,
    endpoint: str,
    latency_seconds: float,
    **kwargs
):
    """
    Log the completion of an API request.
    
    Args:
        logger: Logger instance
        request_id: Unique request identifier
        endpoint: API endpoint that was called
        latency_seconds: Request duration in seconds
        **kwargs: Additional context (confidence_score, refused, etc.)
    """
    logger.info(
        f"{endpoint}_completed",
        extra={
            "request_id": request_id,
            "endpoint": endpoint,
            "latency_seconds": round(latency_seconds, 3),
            **kwargs
        }
    )


def log_request_error(
    logger: logging.Logger,
    request_id: str,
    endpoint: str,
    error: Exception,
    **kwargs
):
    """
    Log an API request error.
    
    Args:
        logger: Logger instance
        request_id: Unique request identifier
        endpoint: API endpoint that failed
        error: Exception that occurred
        **kwargs: Additional context
    """
    logger.error(
        f"{endpoint}_failed",
        extra={
            "request_id": request_id,
            "endpoint": endpoint,
            "error": str(error),
            "error_type": type(error).__name__,
            **kwargs
        },
        exc_info=True
    )