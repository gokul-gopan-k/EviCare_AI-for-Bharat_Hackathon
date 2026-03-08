"""
Clinical Service Module : Resilient LLM generation service for clinical decision support.

This module provides the ClinicalService class responsible for:
- Retrieving clinical guideline chunks
- Generating AI recommendations via Amazon Bedrock
- Computing hybrid confidence scores
- Returning structured clinical recommendations
- Exponential backoff and retries via @with_resilience
- Automated model fallback (Amazon Nova -> Amazon Titan)
- Contextual prompt caching for medical guidelines
- Structured JSON extraction for clinical entities
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

import boto3
import numpy as np
from botocore.exceptions import ClientError

from vector_db.retriever import ClinicalRetriever
from backend.resilience_utils import with_resilience, metrics, resilience_manager


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

LOG_NAME: str = "bedrock_service"
PRIMARY_MODEL_ID: str = "amazon.nova-pro-v1:0"
FALLBACK_MODEL_ID: str = os.getenv("BEDROCK_FALLBACK_GENERATION", "amazon.titan-text-premier-v1:0")
REGION: str = os.getenv("BEDROCK_PRIMARY_REGION", "us-east-1")

logger = logging.getLogger(LOG_NAME)

# ---------------------------------------------------------------------
# Bedrock Service Implementation
# ---------------------------------------------------------------------


class ClinicalService:
    """
    Singleton service responsible for clinical recommendation generation.

    Responsibilities
    ----------------
    - Retrieve relevant clinical guidelines
    - Generate recommendations using Bedrock LLM
    - Compute hybrid reliability scores
    - Return structured recommendation payload
    """

    _instance = None

    def __new__(cls):
        """
        Ensure the service behaves as a Singleton.

        Returns
        -------
        ClinicalService
            The single shared instance of the service.
        """
        if cls._instance is None:
            cls._instance = super(ClinicalService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        Initialize the clinical service.

        This includes:
        - ClinicalRetriever initialization
        - AWS Bedrock runtime client initialization
        """
        if self._initialized:
            return

        try:
            self.retriever = ClinicalRetriever()
            self.client = resilience_manager.client

            self._initialized = True

            logger.info("ClinicalService initialized successfully.")

        except Exception as exc:
            logger.exception("Failed to initialize ClinicalService")
            raise RuntimeError("ClinicalService initialization failed") from exc

    # ------------------------------------------------------------------
    # Hybrid Confidence Score
    # ------------------------------------------------------------------

    def compute_hybrid_confidence(
        self,
        chunks: List[Dict[str, Any]],
        ai_certainty_text: Any,
    ) -> float:
        """
        Compute hybrid reliability score for a recommendation.

        Compute reliability score based on AI certainty.
        
        Handles:
        - Labels: "High" (100), "Medium" (70), "Low" (40)
        - Floats: 0.85 -> 85.0
        """

        if ai_certainty_text is None:
            return 70.0  # Default to Medium if missing

        try:
            #  Handle if LLM sent a float (e.g., 0.85)
            if isinstance(ai_certainty_text, (float, int)):
                # Scale to 0-100. If it's already > 1 (like 85), leave it.
                score = ai_certainty_text if ai_certainty_text > 1 else ai_certainty_text * 100
                return round(float(max(0.0, min(score, 100.0))), 1)
            # 2. Handle if LLM sent a string (e.g., "High")
            certainty_map = {
                "High": 100.0,
                "Medium": 70.0,
                "Low": 40.0,
            }
            # Use the map, default to 70.0 if the string is unknown
            score = certainty_map.get(str(ai_certainty_text).strip().capitalize(), 70.0)
            return float(score)

        except Exception:
            logger.exception("AI certainty conversion failed")
            return 70.0 # Safe fallback

    # ------------------------------------------------------------------
    # JSON Cleaning
    # ------------------------------------------------------------------

    def _clean_llm_json(self, raw_output: str) -> Any:
        """
        Attempt to clean malformed LLM JSON responses.

        Parameters
        ----------
        raw_output : str
            Raw text output from the LLM.

        Returns
        -------
        Any
            Parsed JSON object.
        """

        try:
            cleaned = raw_output.strip()
            cleaned = cleaned.replace("```json", "").replace("```", "")

            start_idx = cleaned.find("[") if "[" in cleaned else cleaned.find("{")
            end_idx = cleaned.rfind("]") if "]" in cleaned else cleaned.rfind("}")

            if start_idx != -1:
                if end_idx == -1 or end_idx < start_idx:
                    cleaned = cleaned[start_idx:] + "}]"
                else:
                    cleaned = cleaned[start_idx : end_idx + 1]

            parsed = json.loads(cleaned)

            if isinstance(parsed, dict) and "recommendations" in parsed:
                return parsed["recommendations"]

            if isinstance(parsed, dict):
                return list(parsed.values())

            return parsed

        except Exception:
            logger.exception("Failed to clean LLM JSON output")
            raise

    # ------------------------------------------------------------------
    # Patient Processing
    # ------------------------------------------------------------------

    @with_resilience(service_type="generation")
    def _invoke_llm(self, model_id: str, prompt: str) -> str:
        """Invokes Bedrock using the resilience manager's active client"""
        # Standardizing on the Converse API for both Nova and Titan
        try:
            # Use the manager's client to benefit from regional failover
            response = resilience_manager.client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"temperature": 0.1, "maxTokens": 1000}
            )
            return response["output"]["message"]["content"][0]["text"]
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            # If the region is down or throttling, switch regions and let the 
            # @with_resilience decorator trigger a retry on the NEW region.
            if error_code in ["ThrottlingException", "ServiceUnavailableException","ModelNotReadyException"]:
                resilience_manager.switch_region()
            raise e 
                

    def process_patient(self, patient_data) -> Dict[str, Any]:
        """
        Generate clinical recommendations for a patient.

        Workflow
        --------
        1. Retrieve guideline chunks
        2. Build context
        3. Invoke Bedrock model
        4. Parse recommendations
        5. Compute reliability scores
        6. Identify risk factors

        Parameters
        ----------
        patient_data : object
            Patient data object containing raw_summary and metrics.

        Returns
        -------
        dict
            Structured recommendation response.
        """

        try:
            # ----------------------------------------------------------
            # 1. Retrieve relevant guideline chunks
            # ----------------------------------------------------------
            chunks = self.retriever.get_relevant_chunks(
                patient_data.raw_summary
            )

            # ----------------------------------------------------------
            # 2. Build context block
            # ----------------------------------------------------------
            context_block = "\n\n".join(
                [
                    (
                        f"Source: {c['metadata'].get('source')} "
                        f"(Section: {c['metadata'].get('section')})\n"
                        f"Content: {c['text']}"
                    )
                    for c in chunks
                ]
            )

            # ----------------------------------------------------------
            # 3. Build LLM prompt
            # ----------------------------------------------------------
            prompt = f"""
        Human: You are a senior clinical architect. Provide 2 distinct medical recommendations based ONLY on the context.
        Output MUST be a valid JSON list and nothing else — no explanation, no markdown, no preamble.

        CRITICAL RULES:
        1. 'reasoning': Explain WHY this matters for this specific patient based on the provided context (max 30 words). Avoid generic phrases like "above target".
        2. 'citation_section': Must be a SHORT quote (max 15 words) from the context that justifies the recommendation.
        3. 'title': A professional clinical action title.
        4. Keys: "title", "strength", "reasoning", "citation_source", "citation_section", "ai_certainty".
        Patient Summary: {patient_data.raw_summary}

        Context:
        {context_block}

        Return only a JSON array like: [{{"title": "...", "strength": "...", ...}}, {{"title": "...", ...}}]
"""

            # ----------------------------------------------------------
            # 4. Invoke Bedrock Model
            # ----------------------------------------------------------
            try:
                try:
                    # Primary Attempt (Nova)
                    raw_output = self._invoke_llm(PRIMARY_MODEL_ID, prompt)
                except Exception as e:
                    # Primary failed all regions/retries, switch to Fallback (Titan)
                    logger.error(f"Primary Model Failure: {str(e)}. Switching to Fallback.")
                    # fallback to tital model
                    raw_output = self._invoke_llm(FALLBACK_MODEL_ID, prompt)
            except Exception as final_e:
                logger.critical(" No model succeeds : All models and regions exhausted.")
                raise final_e

            # ----------------------------------------------------------
            # 5. Parse LLM JSON
            # ----------------------------------------------------------
            try:
                recommendations = self._clean_llm_json(raw_output)

            except Exception:
                logger.error("LLM JSON parsing failed")
                logger.error("Raw LLM output:\n%s", raw_output)

                recommendations = [
                    {
                        "title": "Clinical Review Required",
                        "strength": "Conditional",
                        "reasoning": (
                            "System formatting error. "
                            "Relevant guidelines identified."
                        ),
                        "citation_source": "Multiple Guidelines",
                        "citation_section": "General",
                        "ai_certainty": "Low",
                    }
                ]

            # ----------------------------------------------------------
            # 6. Compute hybrid reliability scores
            # ----------------------------------------------------------
            processed_recommendations = []

            for rec in recommendations[:2]:
                ai_certainty = rec.get("ai_certainty", "Medium")

                reliability = self.compute_hybrid_confidence(
                    chunks,
                    ai_certainty,
                )

                processed_recommendations.append(
                    {
                        **rec,
                        "reliability_score": reliability,
                        "source_chunks": chunks,
                    }
                )

            # ----------------------------------------------------------
            # 7. Risk factor detection
            # ----------------------------------------------------------
            risk_factors: List[str] = []

            hba1c = getattr(patient_data, "hba1c", 0)
            bp_sys = getattr(patient_data, "bp_systolic", 0)
            bp_dias = getattr(patient_data, "bp_diastolic", 0)

            if hba1c and hba1c > 6.5:
                risk_factors.append("Elevated HbA1c")


            if bp_sys >= 140 or bp_dias >= 90:
                risk_factors.append("Hypertension Risk")

            if bp_sys < 90 or bp_dias < 60:
                risk_factors.append("Hypotension Risk")

            # ----------------------------------------------------------
            # 8. Final response payload
            # ----------------------------------------------------------
            return {
                "clinical_summary": patient_data.raw_summary,
                "risk_factors": risk_factors,
                "retrieved_chunks": chunks,
                "recommendations": processed_recommendations,
                "confidence_score": (
                    processed_recommendations[0]["reliability_score"]
                    if processed_recommendations
                    else 0.0
                ),
            }

        except Exception as exc:
            logger.exception("Patient processing failed")
            raise RuntimeError("Clinical processing failed") from exc