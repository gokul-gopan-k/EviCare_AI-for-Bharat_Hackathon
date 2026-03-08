"""
Text Chunking Utility for Document Processing Pipelines.

Provides sentence-aware chunking with configurable overlap and
section detection for structured document ingestion.
"""

from __future__ import annotations

import logging
import re
import subprocess
from typing import Dict, List

import spacy
from spacy.language import Language


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

DEFAULT_TARGET_WORDS: int = 300
DEFAULT_OVERLAP_WORDS: int = 60
SPACY_MODEL_NAME: str = "en_core_web_sm"

LOG_NAME: str = "document_chunker"

HEADING_PATTERN = re.compile(
    r"^((SECTION\s\d+[:\s])|(\d+(\.\d+)*\s)|([A-Z][A-Za-z\s]{3,40}$))",
    re.MULTILINE,
)

ATOMIC_VALUE_PATTERN = re.compile(
    r"\d+\s?(?:mg|mmHg|mmol/L|kg/m2|%)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------

def configure_logger() -> logging.Logger:
    """
    Configure structured logger for the chunking module.

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
# Chunker Service
# ---------------------------------------------------------------------

class Chunker:
    """
    Sentence-aware document chunker.

    Splits large text documents into smaller chunks suitable for
    retrieval pipelines (RAG systems, embeddings, etc.) while
    preserving sentence boundaries and maintaining configurable overlap.
    """

    def __init__(
        self,
        target_words: int = DEFAULT_TARGET_WORDS,
        overlap_words: int = DEFAULT_OVERLAP_WORDS,
    ) -> None:
        """
        Initialize the chunker service.

        Parameters
        ----------
        target_words : int
            Target word count per chunk.
        overlap_words : int
            Number of words to overlap between chunks.
        """
        self.target_words = target_words
        self.overlap_words = overlap_words

        self.nlp = self._initialize_spacy()

    # -----------------------------------------------------------------

    def _initialize_spacy(self) -> Language:
        """
        Safely initialize spaCy pipeline.

        Returns
        -------
        Language
            Loaded spaCy language model.
        """
        try:
            logger.info("Loading spaCy model: %s", SPACY_MODEL_NAME)

            nlp = spacy.load(
                SPACY_MODEL_NAME,
                disable=["ner", "lemmatizer", "tagger", "attribute_ruler"],
            )

            # Enable lightweight sentence segmentation
            nlp.select_pipes(enable=["senter"])

            logger.info("spaCy model loaded successfully")
            return nlp

        except OSError:
            logger.warning("spaCy model not found. Downloading...")

            try:
                subprocess.run(
                    ["python", "-m", "spacy", "download", SPACY_MODEL_NAME],
                    check=True,
                )

                nlp = spacy.load(
                    SPACY_MODEL_NAME,
                    disable=["ner", "lemmatizer", "tagger"],
                )

                nlp.select_pipes(enable=["senter"])

                logger.info("spaCy model downloaded and loaded successfully")
                return nlp

            except Exception as exc:
                logger.exception("Failed to initialize spaCy model")
                raise RuntimeError("spaCy initialization failed") from exc

    # -----------------------------------------------------------------

    def split_by_heading(self, text: str) -> List[Dict[str, List[str]]]:
        """
        Split document into sections using heading patterns.

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        List[Dict[str, List[str]]]
            List of section dictionaries containing title and content.
        """
        lines = text.split("\n")

        sections: List[Dict[str, List[str]]] = []
        current_section: Dict[str, List[str]] = {
            "title": "General",
            "content": [],
        }

        for line in lines:
            stripped_line = line.strip()

            if HEADING_PATTERN.match(stripped_line):
                if current_section["content"]:
                    sections.append(current_section)

                current_section = {
                    "title": stripped_line,
                    "content": [],
                }

            else:
                current_section["content"].append(stripped_line)

        if current_section["content"]:
            sections.append(current_section)

        logger.debug("Detected %d document sections", len(sections))

        return sections

    # -----------------------------------------------------------------

    def create_chunks(self, text_list: List[str]) -> List[str]:
        """
        Generate sentence-aware chunks with overlap.

        Parameters
        ----------
        text_list : List[str]
            List of text segments to combine and chunk.

        Returns
        -------
        List[str]
            Generated text chunks.
        """
        try:
            full_text = " ".join(text_list)

            doc = self.nlp(full_text)
            sentences: List[str] = [sent.text.strip() for sent in doc.sents]

            chunks: List[str] = []
            current_chunk: List[str] = []

            current_word_count = 0

            for sentence in sentences:
                words = sentence.split()
                sentence_word_count = len(words)

                is_atomic = bool(ATOMIC_VALUE_PATTERN.search(sentence))

                if (
                    current_word_count + sentence_word_count
                    > self.target_words
                    and not is_atomic
                ):
                    chunks.append(" ".join(current_chunk))

                    overlap_buffer = self._build_overlap(current_chunk)

                    current_chunk = overlap_buffer + [sentence]

                    current_word_count = (
                        sum(len(s.split()) for s in overlap_buffer)
                        + sentence_word_count
                    )

                else:
                    current_chunk.append(sentence)
                    current_word_count += sentence_word_count

            if current_chunk:
                chunks.append(" ".join(current_chunk))

            logger.info("Generated %d chunks", len(chunks))

            return chunks

        except Exception as exc:
            logger.exception("Chunk generation failed")
            raise RuntimeError("Chunking process failed") from exc

    # -----------------------------------------------------------------

    def _build_overlap(self, chunk_sentences: List[str]) -> List[str]:
        """
        Build sentence-respecting overlap buffer.

        Parameters
        ----------
        chunk_sentences : List[str]
            Sentences in the current chunk.

        Returns
        -------
        List[str]
            Sentences used for overlap.
        """
        overlap_buffer: List[str] = []
        overlap_word_count = 0

        for sentence in reversed(chunk_sentences):
            words = sentence.split()

            overlap_buffer.insert(0, sentence)
            overlap_word_count += len(words)

            if overlap_word_count >= self.overlap_words:
                break

        return overlap_buffer