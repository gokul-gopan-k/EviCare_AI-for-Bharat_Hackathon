"""
ClinicalRetriever: Vector search and Bedrock reranking for clinical guidelines.

- Queries ChromaDB vector store for candidate chunks
- Generates embeddings via Bedrock Titan model
- Reranks top candidates using Bedrock Rerank API
- Multi-region embedding failover (  us-west-2 --> us-east-1)
- Thread-safe TTL caching for search results
- Graceful degradation for reranking failures
- Resilience decorator integration for retries and timeouts
"""

from __future__ import annotations

import json
import logging
import hashlib
import threading
from pathlib import Path
from typing import List, Dict, Any

import boto3
import chromadb
from cachetools import TTLCache

from backend.resilience_utils import with_resilience, metrics

# ---------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------

LOG_NAME: str = "clinical_retriever"
CHROMA_DB_PATH: Path = Path("./vector_db/chunk_data")
COLLECTION_NAME: str = "clinical_guidelines"

EMBEDDING_MODEL_ID: str = "amazon.titan-embed-text-v2:0"
RERANK_MODEL_ARN: str = "arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0"

BEDROCK_REGION_EMBED: str = "us-east-1"
BEDROCK_REGION_RERANK: str = "us-west-2"

BEDROCK_SECONDARY_REGION_EMBED: str = "us-west-2"
BEDROCK_SECONDARY_REGION_RERANK: str = "us-east-1"

DEFAULT_TOP_K: int = 4
DEFAULT_INITIAL_FETCH: int = 15
EMBEDDING_DIMENSIONS: int = 512

# Cache Settings
CACHE_MAX_SIZE: int = 1000
CACHE_TTL_SECONDS: int = 3600  # 1 Hour

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------


def configure_logger() -> logging.Logger:
    """
    Configure structured logger for ClinicalRetriever.
    """
    logger = logging.getLogger(LOG_NAME)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | "
            "%(filename)s:%(lineno)d | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = configure_logger()

# ---------------------------------------------------------------------
# Clinical Retriever
# ---------------------------------------------------------------------


class ClinicalRetriever:
    """
    Retrieves clinically relevant text chunks from a ChromaDB vector store
    and reranks them using AWS Bedrock Titan embeddings and Rerank models.
    """

    def __init__(self, db_path: Path = CHROMA_DB_PATH) -> None:
        """
        Initialize ChromaDB client and AWS Bedrock clients.

        Parameters
        ----------
        db_path : Path
            Path to ChromaDB persistent database.
        """
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_collection(COLLECTION_NAME)

        # Bedrock Runtime with multiregion fallback(for embeddings)
        self.bedrock_clients = {
            BEDROCK_REGION_EMBED: boto3.client("bedrock-runtime", region_name=BEDROCK_REGION_EMBED),
            BEDROCK_SECONDARY_REGION_EMBED: boto3.client("bedrock-runtime", region_name=BEDROCK_SECONDARY_REGION_EMBED)
        }
        self.current_embed_region = BEDROCK_REGION_EMBED
        self.embed_model_id = EMBEDDING_MODEL_ID

        # Bedrock Agent Runtime (for rerank)
        self.bedrock_agent_runtime = boto3.client(
            "bedrock-agent-runtime", region_name=BEDROCK_REGION_RERANK
        )
        self.current_rerank_region = BEDROCK_REGION_RERANK
        self.rerank_model_arn = RERANK_MODEL_ARN

        # Thread-safe Cache
        self._search_cache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL_SECONDS)
        self._cache_lock = threading.Lock()

        logger.info("ClinicalRetriever initialized with multi-region support.")

    # -----------------------------------------------------------------
    # Internal Logic & Resilience
    # -----------------------------------------------------------------

    def _generate_cache_key(self, query_text: str, top_k: int) -> str:
        """Generate a deterministic SHA-256 hash for query caching."""
        key_data = f"{query_text.lower().strip()}:{top_k}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _get_embedding_from_api(self, text: str, region: str) -> List[float]:
        """
        Generate embedding for a text string using AWS Bedrock.

        Parameters
        ----------
        text : str
        region: str
        Returns
        -------
        List[float]
            Vector embedding.
        """
        try:
            body = json.dumps(
                {"inputText": text, 
                 "dimensions": EMBEDDING_DIMENSIONS, 
                 "normalize": True}
            )
            client = self.bedrock_clients[region]

            response = client.invoke_model(
                body=body,
                modelId=self.embed_model_id,
                accept="application/json",
                contentType="application/json",
            )

            embedding = json.loads(response.get("body").read()).get("embedding")
            if not embedding:
                raise ValueError(f"Received empty embedding from Bedrock {region}")
            return embedding
        except Exception as exc:
            logger.exception("Failed to generate embedding for text.")
            raise RuntimeError("Embedding generation failed") from exc


    def _embedding_fallback(self, text: str) -> List[float]:
        """Fallback strategy: Switch regions if primary fails."""
        failover_region = BEDROCK_SECONDARY_REGION_EMBED if self.current_embed_region == BEDROCK_REGION_EMBED else BEDROCK_REGION_EMBED
        logger.warning(f"Embedding failed in {self.current_embed_region}. Failing over to {failover_region}")
        
        try:
            embedding = self._get_embedding_from_api(text, failover_region)
            self.current_embed_region = failover_region  # Stick to working region
            return embedding
        except Exception as e:
            logger.error(f"Critical: Secondary region {failover_region} also failed: {e}")
            raise

    @with_resilience(service_type="embedding")
    def _get_embedding(self, text: str) -> List[float]:
        """Resilient embedding generation with failover."""
        try:
            return self._get_embedding_from_api(text, self.current_embed_region)
        except Exception:
            return self._embedding_fallback(text)

    def _rerank_degradation(self, documents: List[str], metadatas: List[dict], top_k: int) -> List[Dict[str, Any]]:
        """Degradation strategy: Return raw vector results if reranker is down."""
        logger.warning("Reranker unavailable. Applying graceful degradation (raw vector results).")
        
        results = []
        for i in range(min(top_k, len(documents))):
            results.append({
                "text": documents[i],
                "similarity": 0.0,  # Score unknown in degraded mode
                "metadata": {
                    **metadatas[i],
                    "degraded_mode": True,
                    "reranker_bypassed": True
                }
            })
        return results

    @with_resilience(service_type="reranking")
    def _rerank_results(
        self, query_text: str, documents: List[str], metadatas: List[dict], top_k: int
    ) -> List[Dict[str, Any]]:
        """Resiliently rerank candidate chunks using Bedrock Rerank."""
        sources = [
            {
                "type": "INLINE",
                "inlineDocumentSource": {
                    "type": "TEXT",
                    "textDocument": {"text": doc}
                }
            }
            for doc in documents
        ]
        
        response = self.bedrock_agent_runtime.rerank(
            queries=[{"type": "TEXT", "textQuery": {"text": query_text}}],
            sources=sources,
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "numberOfResults": top_k,
                    "modelConfiguration": {"modelArn": self.rerank_model_arn}
                }
            }
        )
        
        final_results = []
        for result in response.get("results", []):
            idx = result.get("index")
            score = result.get("relevanceScore", 0.0)
            if idx is not None and idx < len(documents):
                final_results.append({
                    "text": documents[idx],
                    "similarity": round(score, 4),
                    "metadata": metadatas[idx]
                })
        return final_results
    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def get_relevant_chunks(
        self, 
        query_text: str, 
        top_k: int = DEFAULT_TOP_K, 
        initial_fetch: int = DEFAULT_INITIAL_FETCH
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and rerank relevant chunks for a query.
        Main entry point for retrieving chunks. Uses caching and resilience.
        Parameters
        ----------
        query_text : str
            The user query.
        top_k : int
            Number of final results after rerank.
        initial_fetch : int
            Number of candidate chunks to fetch initially from vector search.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing 'text', 'similarity', and 'metadata'.
        """
        cache_key = self._generate_cache_key(query_text, top_k)
        
        # Check Cache
        with self._cache_lock:
            if cache_key in self._search_cache:
                logger.info("Search cache hit.")
                metrics.increment("cache_hits")
                return self._search_cache[cache_key]

        metrics.increment("cache_misses")


        try:
            # 2. Vector Search
            query_embedding = self._get_embedding(query_text)
            initial_results = self.collection.query(
                query_embeddings=[query_embedding], 
                n_results=initial_fetch
            )

            documents = initial_results.get("documents", [[]])[0]
            metadatas = initial_results.get("metadatas", [[]])[0]

            if not documents:
                return []

            # 3. Rerank (with implicit degradation in decorator)
            try:
                final_results = self._rerank_results(query_text, documents, metadatas, top_k)
            except Exception:
                # Manual trigger of degradation if decorator isn't handling fallback
                final_results = self._rerank_degradation(documents, metadatas, top_k)

            # 4. Update Cache
            with self._cache_lock:
                self._search_cache[cache_key] = final_results
            
            return final_results

        except Exception as e:
            logger.exception(f"Critical failure in retriever: {e}")
            raise RuntimeError("Clinical retrieval system failure") from e