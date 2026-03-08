"""
Vector Database Ingestion for Clinical Guideline Chunks.

- Loads JSON chunk files
- Generates embeddings via AWS Bedrock Titan model
- Stores vectors in ChromaDB Persistent collection
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

import boto3
import chromadb


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

LOG_NAME: str = "vector_ingestion"

COLLECTION_NAME: str = "clinical_guidelines"
EMBEDDING_MODEL: str = "amazon.titan-embed-text-v2:0"
CHROMA_PATH: Path = Path.cwd() / "vector_db/chunk_data"
DATA_DIR: Path = Path("vector_db/chunk_data")

BEDROCK_REGION: str = "us-east-1"
EMBEDDING_DIMENSIONS: int = 512
SLEEP_BETWEEN_CALLS: float = 0.1  # Prevent rate limits
BATCH_LOG_INTERVAL: int = 10  # log every N chunks

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------


def configure_logger() -> logging.Logger:
    """
    Configure structured logger for vector ingestion.
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
# AWS Bedrock Client
# ---------------------------------------------------------------------

def get_bedrock_client() -> boto3.client:
    """
    Initialize the AWS Bedrock client.

    Returns
    -------
    boto3.client
    """
    return boto3.client(service_name="bedrock-runtime", region_name=BEDROCK_REGION)


bedrock_client = get_bedrock_client()


# ---------------------------------------------------------------------
# Embedding Function
# ---------------------------------------------------------------------

def get_bedrock_embedding(text: str) -> List[float]:
    """
    Fetch embedding from AWS Bedrock Titan model.

    Parameters
    ----------
    text : str

    Returns
    -------
    List[float]
        Vector embedding.
    """
    body = json.dumps(
        {
            "inputText": text,
            "dimensions": EMBEDDING_DIMENSIONS,
            "normalize": True,
        }
    )

    try:
        response = bedrock_client.invoke_model(
            body=body,
            modelId=EMBEDDING_MODEL,
            accept="application/json",
            contentType="application/json",
        )

        response_body = json.loads(response.get("body").read())
        embedding = response_body.get("embedding")

        if not embedding:
            raise ValueError("Received empty embedding from Bedrock")

        return embedding

    except Exception as exc:
        logger.exception("Failed to generate embedding from Bedrock")
        raise RuntimeError("Bedrock embedding generation failed") from exc


# ---------------------------------------------------------------------
# ChromaDB Initialization and Ingestion
# ---------------------------------------------------------------------

def initialize_chroma_collection(client: Any, collection_name: str) -> Any:
    """
    Create or reset a ChromaDB collection.

    Parameters
    ----------
    client : API
        ChromaDB client instance.
    collection_name : str

    Returns
    -------
    API
        ChromaDB collection object
    """
    try:
        try:
            client.delete_collection(collection_name)
            logger.info("Deleted existing collection: %s", collection_name)
        except Exception:
            pass

        collection = client.create_collection(name=collection_name)
        logger.info("Created collection: %s", collection_name)
        return collection
    except Exception as exc:
        logger.exception("Failed to initialize ChromaDB collection")
        raise RuntimeError("ChromaDB collection initialization failed") from exc


# ---------------------------------------------------------------------
# Load JSON Chunk Files
# ---------------------------------------------------------------------

def load_chunks_from_dir(data_dir: Path) -> List[Dict[str, Any]]:
    """
    Load JSON chunk data from a directory.

    Parameters
    ----------
    data_dir : Path

    Returns
    -------
    List[Dict[str, Any]]
    """
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir.resolve())
        return []

    all_chunks: List[Dict[str, Any]] = []

    json_files = list(data_dir.glob("*.json"))

    if not json_files:
        logger.warning("No JSON files found in %s", data_dir)

    for path in json_files:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                all_chunks.extend(data)
            logger.info("Loaded %d chunks from %s", len(data), path.name)
        except Exception as exc:
            logger.error("Failed to load %s: %s", path.name, exc)

    return all_chunks


# ---------------------------------------------------------------------
# Ingest Chunks into ChromaDB
# ---------------------------------------------------------------------

def ingest_chunks_to_chroma(all_chunks: List[Dict[str, Any]]) -> None:
    """
    Generate embeddings and add them to ChromaDB.

    Parameters
    ----------
    all_chunks : List[Dict[str, Any]]
    """
    if not all_chunks:
        logger.error("No chunks to ingest.")
        return

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = initialize_chroma_collection(client, COLLECTION_NAME)

    embeddings: List[List[float]] = []
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    logger.info("Generating embeddings for %d chunks", len(all_chunks))

    for i, chunk in enumerate(all_chunks):
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})

        if not text:
            continue

        try:
            vector = get_bedrock_embedding(text)

            embeddings.append(vector)
            texts.append(text)
            metadatas.append(metadata)
            ids.append(f"id_{i}")

            if (i + 1) % BATCH_LOG_INTERVAL == 0:
                logger.info("Processed %d/%d chunks", i + 1, len(all_chunks))

            time.sleep(SLEEP_BETWEEN_CALLS)

        except Exception as exc:
            logger.error("Failed to embed chunk %d: %s", i, exc)

    if not embeddings:
        logger.error("No embeddings generated. Aborting ingestion.")
        return

    logger.info("Adding embeddings to ChromaDB collection")
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    logger.info("ChromaDB ingestion complete.")


# ---------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------

def main() -> None:
    """
    Main function to load, embed, and ingest chunks into ChromaDB.
    """
    all_chunks = load_chunks_from_dir(DATA_DIR)
    ingest_chunks_to_chroma(all_chunks)


# ---------------------------------------------------------------------
# CLI Runner
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()