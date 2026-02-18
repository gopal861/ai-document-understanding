# app/main.py
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import uuid

from app.api.routes import router, load_document_registry
from app.observability.logger import setup_logging, get_logger
from app.observability.metrics import metrics_tracker
from app.observability.posthog_client import posthog_client   # NEW

# Initialize logging FIRST
setup_logging(log_level="INFO")
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Document Understanding API",
    description="Multi-document RAG system with similarity-based refusal",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all HTTP requests with latency tracking
    AND record production metrics safely
    AND register tracing in PostHog.
    """

    # Generate unique request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # NEW: Register request identity in PostHog (SAFE)
    posthog_client.identify_user(
        distinct_id=request_id,
        properties={
            "entry_point": request.url.path,
            "method": request.method,
        },
    )

    # Log request start
    logger.info(
        "request_started",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else None
        }
    )

    start_time = time.time()

    try:

        response = await call_next(request)

        latency = time.time() - start_time

        # Record success metrics
        metrics_tracker.record_success(latency)

        # Log request completion
        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_seconds": round(latency, 3)
            }
        )

        return response

    except Exception as e:

        latency = time.time() - start_time

        # Record failure metrics
        metrics_tracker.record_failure()

        # Track error in PostHog (SAFE)
        posthog_client.track_error(
            distinct_id=request_id,
            error_type=type(e).__name__,
            error_message=str(e),
            endpoint=request.url.path,
        )

        # Log request error
        logger.error(
            "request_failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "latency_seconds": round(latency, 3),
                "error": str(e),
                "error_type": type(e).__name__
            },
            exc_info=True
        )

        raise


# Include API routes
app.include_router(router)


@app.on_event("startup")
async def startup_event():

    load_document_registry()

    logger.info("application_startup", extra={"version": "1.0.0"})

    if not os.getenv("OPENAI_API_KEY"):

        logger.warning(
            "missing_api_key",
            extra={
                "warning_detail":
                "OPENAI_API_KEY not set. API calls will fail."
            }
        )

    print("=" * 50)
    print("AI Document Understanding API Started")
    print("=" * 50)
    print("Endpoints:")
    print("  POST /upload           - Upload a document")
    print("  POST /ask              - Ask a question")
    print("  GET  /documents        - List all documents")
    print("  DELETE /documents/{id} - Delete a document")
    print("  GET  /health           - Health check")
    print("  GET  /metrics          - System metrics")
    print("=" * 50)
    print("Logs: logs/app.log (JSON format)")
    print("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():

    logger.info("application_shutdown")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):

    request_id = getattr(request.state, "request_id", "unknown")

    logger.error(
        "unhandled_exception",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "error": str(exc),
            "error_type": type(exc).__name__
        },
        exc_info=True
    )

    # Track global exception in PostHog
    posthog_client.track_error(
        distinct_id=request_id,
        error_type=type(exc).__name__,
        error_message=str(exc),
        endpoint=request.url.path,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal error occurred. Please try again.",
            "request_id": request_id,
            "error_type": type(exc).__name__
        }
    )


@app.get("/")
async def root():

    return {
        "message": "AI Document Understanding API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }
