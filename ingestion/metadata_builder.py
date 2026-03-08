"""
Metadata Builder for Clinical Guideline Chunks.

Generates structured metadata entries and deterministic IDs
for downstream indexing and retrieval pipelines.
"""

from __future__ import annotations

import logging
import re
from typing import Dict

from pydantic import BaseModel


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

LOG_NAME: str = "metadata_builder"

MAX_SECTION_LENGTH: int = 20

SANITIZE_PATTERN = re.compile(r"[^a-z0-9]")


# ---------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------

def configure_logger() -> logging.Logger:
    """
    Configure structured logger for the metadata builder.

    Returns
    -------
    logging.Logger
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
# Data Models
# ---------------------------------------------------------------------

class ChunkMetadata(BaseModel):
    """
    Metadata describing the context of a text chunk.
    """

    source: str
    guideline: str
    condition: str
    section: str
    country_context: str


class ChunkEntry(BaseModel):
    """
    Structured chunk entry for indexing or storage.
    """

    id: str
    text: str
    metadata: ChunkMetadata


# ---------------------------------------------------------------------
# Metadata Builder Service
# ---------------------------------------------------------------------

class MetadataBuilder:
    """
    Service responsible for building structured metadata entries
    with deterministic IDs for document chunks.
    """

    def __init__(self, source: str, guideline: str, country: str) -> None:
        """
        Initialize metadata builder.

        Parameters
        ----------
        source : str
            Document source (e.g., ADA, WHO).
        guideline : str
            Guideline name.
        country : str
            Country context for guideline.
        """
        try:
            self.source = source
            self.guideline = guideline
            self.country = country

            self.id_registry: Dict[str, int] = {}

            logger.info(
                "MetadataBuilder initialized | source=%s guideline=%s country=%s",
                source,
                guideline,
                country,
            )

        except Exception as exc:
            logger.exception("MetadataBuilder initialization failed")
            raise RuntimeError("MetadataBuilder initialization failed") from exc

    # -----------------------------------------------------------------

    def _sanitize(self, value: str) -> str:
        """
        Normalize strings for ID generation.

        Parameters
        ----------
        value : str

        Returns
        -------
        str
        """
        return SANITIZE_PATTERN.sub("_", value.lower())

    # -----------------------------------------------------------------

    def format_id(self, condition: str, section: str) -> str:
        """
        Generate deterministic ID for a chunk.

        Parameters
        ----------
        condition : str
        section : str

        Returns
        -------
        str
        """
        try:
            clean_source = self._sanitize(self.source)

            clean_condition = self._sanitize(condition)

            clean_section = self._sanitize(section)[:MAX_SECTION_LENGTH]

            base_id = f"{clean_source}_{clean_condition}_{clean_section}"

            self.id_registry[base_id] = self.id_registry.get(base_id, 0) + 1

            generated_id = f"{base_id}_{self.id_registry[base_id]:02d}"

            logger.debug("Generated chunk ID: %s", generated_id)

            return generated_id

        except Exception as exc:
            logger.exception("ID generation failed")
            raise RuntimeError("Chunk ID generation failed") from exc

    # -----------------------------------------------------------------

    def build_entry(
        self,
        text: str,
        condition: str,
        section_title: str,
    ) -> ChunkEntry:
        """
        Build structured metadata entry for a text chunk.

        Parameters
        ----------
        text : str
        condition : str
        section_title : str

        Returns
        -------
        ChunkEntry
        """
        try:
            chunk_id = self.format_id(condition, section_title)

            metadata = ChunkMetadata(
                source=self.source,
                guideline=self.guideline,
                condition=condition,
                section=section_title,
                country_context=self.country,
            )

            entry = ChunkEntry(
                id=chunk_id,
                text=text,
                metadata=metadata,
            )

            return entry

        except Exception as exc:
            logger.exception("Failed to build metadata entry")
            raise RuntimeError("Metadata entry creation failed") from exc