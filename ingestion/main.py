"""
Clinical Guideline Processing Pipeline.

This pipeline performs the following stages:
1. PDF text extraction
2. Clinical text cleaning
3. Section detection and filtering
4. Sentence-aware chunking
5. Metadata enrichment
6. JSON dataset generation
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

from pdf_extractor import PDFExtractor
from cleaner import TextCleaner
from section_detector import SectionDetector
from chunker import Chunker
from metadata_builder import MetadataBuilder


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

LOG_NAME: str = "guideline_pipeline"

MIN_CHUNK_WORDS: int = 20
DEFAULT_TARGET_WORDS: int = 300
DEFAULT_OVERLAP_WORDS: int = 60


# ---------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------

def configure_logger() -> logging.Logger:
    """
    Configure structured pipeline logger.

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
# Pipeline Service
# ---------------------------------------------------------------------

class GuidelinePipeline:
    """
    End-to-end pipeline for transforming clinical guideline PDFs
    into structured JSON chunks for downstream AI systems.
    """

    def __init__(self, source: str, guideline: str, country: str) -> None:
        """
        Initialize pipeline dependencies.

        Parameters
        ----------
        source : str
        guideline : str
        country : str
        """
        try:
            logger.info("Initializing pipeline services")

            self.cleaner = TextCleaner()
            self.detector = SectionDetector()
            self.chunker = Chunker(
                target_words=DEFAULT_TARGET_WORDS,
                overlap_words=DEFAULT_OVERLAP_WORDS,
            )

            self.meta_builder = MetadataBuilder(source, guideline, country)

            logger.info("Pipeline services initialized successfully")

        except Exception as exc:
            logger.exception("Pipeline initialization failed")
            raise RuntimeError("Pipeline initialization failed") from exc

    # -----------------------------------------------------------------

    def process_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Process a guideline PDF into structured metadata chunks.

        Parameters
        ----------
        pdf_path : Path

        Returns
        -------
        List[Dict[str, Any]]
        """
        raw_text = self._extract_text(pdf_path)

        clean_text = self.cleaner.clean(raw_text)

        sections = self.chunker.split_by_heading(clean_text)

        logger.info("Detected %d sections", len(sections))

        final_output: List[Dict[str, Any]] = []

        for section in sections:
            section_title: str = section["title"]
            section_text: str = " ".join(section["content"])

            if not self.detector.is_relevant(section_text, section_title):
                continue

            condition = self.detector.determine_condition(section_text)

            chunks = self.chunker.create_chunks(section["content"])

            for chunk in chunks:

                if len(chunk.split()) < MIN_CHUNK_WORDS:
                    continue

                entry = self.meta_builder.build_entry(
                    chunk,
                    condition,
                    section_title,
                )

                final_output.append(entry)

        logger.info("Generated %d structured chunks", len(final_output))

        return final_output

    # -----------------------------------------------------------------

    def _extract_text(self, pdf_path: Path) -> str:
        """
        Extract raw text from PDF.

        Parameters
        ----------
        pdf_path : Path

        Returns
        -------
        str
        """
        try:
            extractor = PDFExtractor(str(pdf_path))

            raw_text = extractor.extract_raw_text()

            if not raw_text or not raw_text.strip():
                raise ValueError("PDF extraction produced empty text")

            logger.info("Extracted %d characters", len(raw_text))

            return raw_text

        except Exception as exc:
            logger.exception("PDF extraction failed")
            raise RuntimeError("PDF extraction failed") from exc


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

def validate_pdf_path(pdf_path: Path) -> None:
    """
    Validate input PDF file.

    Parameters
    ----------
    pdf_path : Path
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Invalid file type: {pdf_path}")


def save_output(data: List[Dict[str, Any]], pdf_path: Path) -> Path:
    """
    Save pipeline output to JSON.

    Parameters
    ----------
    data : List[Dict[str, Any]]
    pdf_path : Path

    Returns
    -------
    Path
    """
    output_path = pdf_path.with_suffix(".json")

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    logger.info("Output saved to %s", output_path)

    return output_path


# ---------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------

def main(pdf_path: str, source: str, guideline: str, country: str) -> None:
    """
    Main CLI entry point.

    Parameters
    ----------
    pdf_path : str
    source : str
    guideline : str
    country : str
    """
    try:
        path = Path(pdf_path)

        validate_pdf_path(path)

        pipeline = GuidelinePipeline(source, guideline, country)

        output_data = pipeline.process_pdf(path)

        save_output(output_data, path)

        logger.info("Pipeline completed successfully")

    except Exception as exc:
        logger.error("Pipeline execution failed: %s", exc)
        sys.exit(1)


# ---------------------------------------------------------------------
# CLI Runner
# ---------------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 5:
        print(
            "Usage:\n"
            "python main.py <file.pdf> <source> <guideline_name> <country>"
        )
        sys.exit(1)

    main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
    )