"""
RETINA-Q — FastAPI Application Entrypoint

Hybrid Quantum-Classical Multi-Modal Retinal Disease Diagnosis System
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import settings
from app.routes import predict, segment
from app.schemas.responses import HealthResponse

# ──────────────────────────────────────────────────────────────
# Configure logging
# ──────────────────────────────────────────────────────────────
logger.add(
    "logs/retina_q.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

# ──────────────────────────────────────────────────────────────
# Create FastAPI application
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="RETINA-Q API",
    description=(
        "Hybrid Quantum-Classical Multi-Modal Retinal Disease Diagnosis System. "
        "Provides OCT classification, fundus classification, macular segmentation, "
        "and AI explainability through Grad-CAM and feature importance mapping."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ──────────────────────────────────────────────────────────────
# CORS Middleware
# ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────
# Register Routers
# ──────────────────────────────────────────────────────────────
app.include_router(predict.router)
app.include_router(segment.router)


# ──────────────────────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────────────────────
@app.get("/", response_model=HealthResponse, tags=["Health"])
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse()


# ──────────────────────────────────────────────────────────────
# Startup / Shutdown Events
# ──────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("RETINA-Q API starting up")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Allowed origins: {settings.allowed_origins}")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("RETINA-Q API shutting down")
