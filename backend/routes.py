"""
API Routes for Clinical Recommendation Service.

This module defines FastAPI endpoints responsible for:
- Receiving patient data
- Calling the clinical recommendation engine
- Returning structured recommendations
"""

import json
import logging
import time
import traceback
from typing import Optional
import threading
import os 
from fastapi import APIRouter, HTTPException

from backend.resilience_utils import metrics, resilience_manager
from backend.schemas import PatientData, RecommendationResponse
from backend.services import ClinicalService

# ------------------------------------------------------------------
# Logger Configuration
# ------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

DEBUG_LOG_PATH = os.path.join(os.getcwd(), "debug.log")

# ------------------------------------------------------------------
# Router Initialization
# ------------------------------------------------------------------

router = APIRouter()

# Singleton service instance
_service_instance: Optional[ClinicalService] = None
_service_lock = threading.Lock()


# ------------------------------------------------------------------
# Debug Logging Utility
# ------------------------------------------------------------------


def _write_debug_log(payload: dict) -> None:
    """
    Write structured debug logs to a file.

    Parameters
    ----------
    payload : dict
        Log payload to write.
    """
    try:
        payload["timestamp"] = time.time() * 1000

        with open(DEBUG_LOG_PATH, "a") as log_file:
            log_file.write(json.dumps(payload) + "\n")

    except Exception:
        logger.exception("Failed to write debug log")


# ------------------------------------------------------------------
# Service Getter
# ------------------------------------------------------------------


def get_service() -> ClinicalService:
    """
    Retrieve the singleton instance of ClinicalService.

    Returns
    -------
    ClinicalService
        Initialized clinical recommendation service.
    """

    global _service_instance

    try:
        if _service_instance is None:
            with _service_lock:  # Only one request can enter this block at a time
                if _service_instance is None:
                    _service_instance = ClinicalService()
        return _service_instance

    except Exception as exc:
        logger.exception("ClinicalService initialization failed")

        raise HTTPException(
            status_code=500,
            detail="Clinical service initialization failed",
        ) from exc


# ------------------------------------------------------------------
# Recommendation Endpoint
# ------------------------------------------------------------------


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    tags=["Clinical Recommendations"],
)
async def recommend(data: PatientData):
    """
    Generate clinical recommendations for a patient.

    Parameters
    ----------
    data : PatientData
        Patient input data containing summary and clinical values.

    Returns
    -------
    RecommendationResponse
        AI-generated clinical recommendations.
    """

    # --------------------------------------------------------------
    # Request Received Log
    # --------------------------------------------------------------

    _write_debug_log(
        {
            "hypothesisId": "H4",
            "location": "routes.py:recommend",
            "message": "request_received",
            "data": {
                "keys": list(data.model_dump().keys()),
                "raw_summary_len": (
                    len(data.raw_summary) if data.raw_summary else 0
                ),
            },
        }
    )

    try:
        # Retrieve service
        service = get_service()

        # Process patient data
        result = service.process_patient(data)

        return result

    except Exception as exc:
        logger.exception("Recommendation processing failed")

        # ----------------------------------------------------------
        # Debug Logging for Failure
        # ----------------------------------------------------------

        _write_debug_log(
            {
                "hypothesisId": "H1-H5",
                "location": "routes.py:recommend",
                "message": "exception_caught",
                "data": {
                    "type": type(exc).__name__,
                    "detail": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to generate clinical recommendations",
        ) from exc
    
@router.get("/health", tags=["System Health"])
async def get_health():
    """
    Returns the health status and resilience metrics of the clinical service.
    """
    service = get_service()
    
    # Check dependencies
    try:
        # Check if Bedrock client is responsive
        resilience_manager.client.list_foundation_models(maxResults=1)
        bedrock_status = "connected"
    except Exception as e:
        logger.warning(f"Bedrock health check failed: {str(e)}")
        bedrock_status = "error"

    return {
        "status": "healthy" if bedrock_status == "connected" else "degraded",
        "regions": {
            "current_generation": resilience_manager.regions[resilience_manager.current_region_index],
            "current_embedding": service.retriever.current_embed_region,
            "current_rerank": service.retriever.current_rerank_region
        },
        "resilience_metrics": metrics.get_all(),
        "timestamp": time.time()
    }