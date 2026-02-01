# app/main.py
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import uuid

from app.api.routes import router
from app.observability.logger import setup_logging, get_logger

# Initialize logging FIRST (before anything else)
setup_logging(log_level="INFO")
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Document Understanding API",
    description="Multi-document RAG system with similarity-based refusal",
    version="1.0.0"
)

# CORS middleware (allows UI to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all HTTP requests with latency tracking.
    
    Adds request_id to all requests for tracing.
    """
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
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
    
    # Track latency
    start_time = time.time()
    
    try:
        response = await call_next(request)
        latency = time.time() - start_time
        
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
    """
    Validate environment and log startup information.
    """
    logger.info("application_startup", extra={"version": "1.0.0"})
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning(
            "missing_api_key",
            extra={"message": "OPENAI_API_KEY not set. API calls will fail."}
        )
        print("WARNING: OPENAI_API_KEY not set. Set it with: export OPENAI_API_KEY='sk-...'")
    
    print("=" * 50)
    print("AI Document Understanding API Started")
    print("=" * 50)
    print("Endpoints:")
    print("  POST /upload           - Upload a document")
    print("  POST /ask              - Ask a question")
    print("  GET  /documents        - List all documents")
    print("  DELETE /documents/{id} - Delete a document")
    print("  GET  /health           - Health check")
    print("=" * 50)
    print("Logs: logs/app.log (JSON format)")
    print("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """Log application shutdown."""
    logger.info("application_shutdown")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all exception handler to prevent 500 errors from crashing the app.
    """
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
    """
    Root endpoint - API information.
    """
    return {
        "message": "AI Document Understanding API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }