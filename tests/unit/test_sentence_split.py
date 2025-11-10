"""
Unit tests for sentence_split module
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from sentence_split import split_sentences

class TestSplitSentences:
    """Tests for split_sentences function"""

    def test_single_sentence(self):
        """Should handle single sentence"""
        text = "Esta es una oración."
        result = split_sentences(text)
        assert len(result) == 1
        assert result[0] == "Esta es una oración."

    def test_multiple_sentences(self):
        """Should split multiple sentences"""
        text = "Primera oración. Segunda oración. Tercera oración."
        result = split_sentences(text)
        assert len(result) == 3

    def test_preserves_abbreviations(self):
        """Should not split on abbreviations"""
        text = "El Dr. García coordinará la reunión."
        result = split_sentences(text)
        assert len(result) == 1

    def test_handles_newlines(self):
        """Should handle newlines appropriately"""
        text = "Primera línea.\nSegunda línea."
        result = split_sentences(text)
        assert len(result) >= 1

    def test_empty_string(self):
        """Should handle empty string"""
        result = split_sentences("")
        assert len(result) == 0

    def test_spanish_punctuation(self):
        """Should handle Spanish punctuation"""
        text = "¿Quién coordinará? Juan coordinará."
        result = split_sentences(text)
        assert len(result) >= 1

    def test_strips_whitespace_from_sentences(self):
        """Should strip whitespace from each sentence"""
        text = "   Primera oración.   Segunda oración.   "
        result = split_sentences(text)
        for sent in result:
            assert sent == sent.strip()

    def test_filters_empty_sentences(self):
        """Should filter out empty sentences"""
        text = "Oración válida. . . Otra oración válida."
        result = split_sentences(text)
        for sent in result:
            assert len(sent) > 0
