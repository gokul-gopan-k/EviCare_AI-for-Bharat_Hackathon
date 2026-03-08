"""
Clinical Text Cleaning Utility.

Removes administrative noise and truncates non-clinical sections
from medical guideline documents prior to downstream processing.
"""

from __future__ import annotations

import logging
import re
from typing import List, Pattern


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

LOG_NAME: str = "text_cleaner"

JUNK_PATTERNS: List[str] = [
    r"Page \d+ of \d+",
    r"^\d+$",
    r"©.*All rights reserved",
    r"ISBN:? \d+",
    r"http[s]?://\S+",
]

END_SECTION_MARKERS: List[str] = [
    r"\nReferences\n",
    r"\nBibliography\n",
    r"\nAnnexures?\s?\d?",
    r"\nResources\n",
    r"\nFurther Reading",
]


# ---------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------

def configure_logger() -> logging.Logger:
    """
    Configure structured logger for the text cleaner.

    Returns
    -------
    logging.Logger
        Configured logger instance.
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
# Text Cleaner Service
# ---------------------------------------------------------------------

class TextCleaner:
    """
    Utility class for removing non-clinical noise from documents.

    This cleaner removes:
    - pagination artifacts
    - administrative copyright text
    - URLs and metadata
    - bibliography and reference sections
    """

    def __init__(self) -> None:
        """
        Initialize cleaner and compile regex patterns.
        """
        try:
            self.junk_patterns: List[Pattern[str]] = [
                re.compile(pattern, re.IGNORECASE) for pattern in JUNK_PATTERNS
            ]

            self.end_markers: List[Pattern[str]] = [
                re.compile(pattern, re.IGNORECASE) for pattern in END_SECTION_MARKERS
            ]

            logger.info("TextCleaner initialized with %d junk patterns", len(self.junk_patterns))

        except re.error as exc:
            logger.exception("Regex compilation failed during TextCleaner initialization")
            raise RuntimeError("TextCleaner initialization failed") from exc

    # -----------------------------------------------------------------

    def clean(self, text: str) -> str:
        """
        Clean clinical document text.

        Processing steps:
        1. Remove trailing bibliography/reference sections
        2. Remove administrative junk lines
        3. Normalize whitespace

        Parameters
        ----------
        text : str
            Raw document text.

        Returns
        -------
        str
            Cleaned clinical text.
        """
        try:
            truncated_text = self._truncate_end_sections(text)

            cleaned_lines = self._remove_noise_lines(truncated_text)

            result = "\n".join(cleaned_lines)

            logger.debug("Text cleaning complete. Output length: %d", len(result))

            return result

        except Exception as exc:
            logger.exception("Text cleaning failed")
            raise RuntimeError("Text cleaning process failed") from exc

    # -----------------------------------------------------------------

    def _truncate_end_sections(self, text: str) -> str:
        """
        Remove reference/bibliography sections from document.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
        """
        for marker in self.end_markers:
            match = marker.search(text)

            if match:
                logger.info("End section detected: truncating document")
                return text[: match.start()]

        return text

    # -----------------------------------------------------------------

    def _remove_noise_lines(self, text: str) -> List[str]:
        """
        Remove administrative and non-clinical lines.

        Parameters
        ----------
        text : str

        Returns
        -------
        List[str]
            Clean lines ready for downstream processing.
        """
        lines = text.splitlines()

        cleaned_lines: List[str] = []

        for line in lines:
            stripped_line = line.strip()

            if not stripped_line:
                continue

            if any(pattern.search(stripped_line) for pattern in self.junk_patterns):
                logger.debug("Removed junk line: %s", stripped_line)
                continue

            cleaned_lines.append(stripped_line)

        logger.info("Retained %d lines after cleaning", len(cleaned_lines))

        return cleaned_lines