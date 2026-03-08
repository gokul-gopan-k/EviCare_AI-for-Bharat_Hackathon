"""
Clinical Section Detector.

Determines whether a section of text is relevant for clinical processing
and identifies its condition context (e.g., Diabetes, Hypertension).
"""

from __future__ import annotations

import logging
import re
from typing import List, Dict

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

LOG_NAME: str = "section_detector"

CONDITION_KEYWORDS: Dict[str, List[str]] = {
    "Diabetes": [r"diabet", r"glucose", r"hba1c", r"insulin", r"glycaemic", r"hyperglyc"],
    "Hypertension": [r"hypertension", r"blood pressure", r"systolic", r"diastolic", r"mmhg"],
    "Diagnostic": [r"threshold", r"diagnostic", r"criteria", r"cut-off", r"screening", r"test"],
    "Treatment": [r"metformin", r"sulfonylurea", r"gliclazide", r"dosage", r"medication", r"first-line"],
}

IGNORE_TITLES: List[str] = [
    r"preface",
    r"acknowledgement",
    r"abbreviation",
    r"disclaimer",
    r"copyright",
]

SHORT_SECTION_WORD_THRESHOLD: int = 100

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------


def configure_logger() -> logging.Logger:
    """
    Configure structured logger for the section detector.

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
# Section Detector
# ---------------------------------------------------------------------


class SectionDetector:
    """
    Detects whether a text section is clinically relevant and determines its condition.
    """

    def __init__(self) -> None:
        """
        Initialize SectionDetector with precompiled regex patterns.
        """
        try:
            # Precompile regex for performance
            self.keywords: Dict[str, List[re.Pattern]] = {
                category: [re.compile(kw, re.IGNORECASE) for kw in kws]
                for category, kws in CONDITION_KEYWORDS.items()
            }

            self.ignore_titles: List[re.Pattern] = [
                re.compile(title, re.IGNORECASE) for title in IGNORE_TITLES
            ]

            logger.info("SectionDetector initialized with keywords and ignore titles")

        except re.error as exc:
            logger.exception("Regex compilation failed during SectionDetector initialization")
            raise RuntimeError("SectionDetector initialization failed") from exc

    # -----------------------------------------------------------------

    def is_relevant(self, text: str, title: str) -> bool:
        """
        Determine if a section is clinically relevant.

        Parameters
        ----------
        text : str
            Section text content.
        title : str
            Section title.

        Returns
        -------
        bool
            True if section is relevant, False otherwise.
        """
        lower_text = text.lower()

        # 1️⃣ Skip administrative sections
        if any(pattern.search(title) for pattern in self.ignore_titles):
            logger.debug("Section skipped due to ignore title: %s", title)
            return False

        # 2️⃣ Check for condition mentions
        has_condition = any(
            any(pattern.search(lower_text) for pattern in self.keywords[cond])
            for cond in ["Diabetes", "Hypertension"]
        )

        # 3️⃣ Check for action mentions
        has_action = any(
            any(pattern.search(lower_text) for pattern in self.keywords[act])
            for act in ["Diagnostic", "Treatment"]
        )

        # 4️⃣ Apply leniency for short sections (tables, bullets)
        word_count = len(text.split())
        if word_count < SHORT_SECTION_WORD_THRESHOLD:
            return has_condition

        return has_condition and has_action

    # -----------------------------------------------------------------

    def determine_condition(self, text: str) -> str:
        """
        Determine the primary condition(s) referenced in a section.

        Parameters
        ----------
        text : str
            Section text content.

        Returns
        -------
        str
            Condition label: "Diabetes", "Hypertension", "Diabetes_Hypertension", or "General"
        """
        lower_text = text.lower()

        is_dm = any(pattern.search(lower_text) for pattern in self.keywords["Diabetes"])
        is_htn = any(pattern.search(lower_text) for pattern in self.keywords["Hypertension"])

        if is_dm and is_htn:
            return "Diabetes_Hypertension"
        elif is_dm:
            return "Diabetes"
        elif is_htn:
            return "Hypertension"
        else:
            return "General"