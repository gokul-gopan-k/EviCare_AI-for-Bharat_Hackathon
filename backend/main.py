"""
FastAPI Application Entry Point

This module initializes the EviCare backend application.

Responsibilities
---------------
- Configure FastAPI app
- Setup middleware (CORS)
- Mount static file serving
- Register API routers
- Provide health check endpoint
"""

from dotenv import load_dotenv 

load_dotenv() 

import logging
import os
from contextlib import asynccontextmanager


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routes import router,  get_service

# ------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

APP_NAME = "EviCare Backend"
APP_VERSION = "1.0.0"

STATIC_FOLDER = os.path.join(os.getcwd(), "data", "pdfs")

# ------------------------------------------------------------------
# Application Lifespan
# ------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.

    Handles startup and shutdown events.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance.
    """

    # ------------------------------
    # Startup Logic
    # ------------------------------
    try:
        logger.info("Starting EviCare backend service")
        get_service() 
        logger.info("ClinicalService warmed and ready")
        logger.info("Bedrock API initialized")

        # Ensure static directory exists
        os.makedirs(STATIC_FOLDER, exist_ok=True)

        logger.info("Static PDF directory ready: %s", STATIC_FOLDER)

    except Exception:
        logger.exception("Application startup failed")
        raise

    yield

    # ------------------------------
    # Shutdown Logic
    # ------------------------------
    logger.info("Shutting down EviCare backend service")


# ------------------------------------------------------------------
# FastAPI App Initialization
# ------------------------------------------------------------------

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    lifespan=lifespan,
)

# ------------------------------------------------------------------
# Middleware Configuration
# ------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Static File Mounting
# ------------------------------------------------------------------

try:
    app.mount(
        "/static",
        StaticFiles(directory=STATIC_FOLDER),
        name="static",
    )
    logger.info("Static file route mounted at /static")

except Exception:
    logger.exception("Failed to mount static directory")

# ------------------------------------------------------------------
# Router Registration
# ------------------------------------------------------------------

app.include_router(router, prefix="/api")

# ADD THE DECORATOR HERE:
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "resilience_metrics": {"fallback_count": 0} 
    }

