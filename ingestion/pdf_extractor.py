"""
PDF Text Extraction Service.

Provides safe and efficient extraction of raw text from PDF documents
using PyMuPDF (fitz). Designed for use in document ingestion pipelines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import fitz  # PyMuPDF


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

LOG_NAME: str = "pdf_extractor"
TEXT_MODE: str = "text"


# ---------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------

def configure_logger() -> logging.Logger:
    """
    Configure structured logger for the PDF extractor.

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
# PDF Extraction Service
# ---------------------------------------------------------------------

class PDFExtractor:
    """
    Service responsible for extracting raw text from PDF documents.
    """

    def __init__(self, file_path: str) -> None:
        """
        Initialize PDF extractor.

        Parameters
        ----------
        file_path : str
            Path to the PDF file.
        """
        try:
            self.file_path = Path(file_path)

            if not self.file_path.exists():
                raise FileNotFoundError(f"PDF not found: {file_path}")

            logger.info("PDFExtractor initialized for %s", self.file_path)

        except Exception as exc:
            logger.exception("Failed to initialize PDFExtractor")
            raise RuntimeError("PDFExtractor initialization failed") from exc

    # -----------------------------------------------------------------

    def extract_raw_text(self) -> str:
        """
        Extract raw text from the PDF document.

        Returns
        -------
        str
            Extracted text content from all pages.
        """
        try:
            full_text: List[str] = []

            with fitz.open(self.file_path) as document:

                page_count = len(document)
                logger.info("Opened PDF with %d pages", page_count)

                for page_number, page in enumerate(document, start=1):

                    page_text = page.get_text(TEXT_MODE)

                    if page_text:
                        full_text.append(page_text)

                    logger.debug(
                        "Extracted text from page %d",
                        page_number,
                    )

            combined_text = "\n".join(full_text)

            logger.info(
                "Extraction complete | characters=%d",
                len(combined_text),
            )

            return combined_text

        except Exception as exc:
            logger.exception(
                "Failed to extract PDF text | file=%s",
                self.file_path,
            )
            raise RuntimeError("PDF text extraction failed") from exc