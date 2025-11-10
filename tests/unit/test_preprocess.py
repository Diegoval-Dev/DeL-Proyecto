"""
Unit tests for preprocess module
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from preprocess import clean_text

class TestCleanText:
    """Tests for clean_text function"""

    def test_removes_carriage_returns(self):
        """Should replace \\r with \\n"""
        text = "Hola\r\nMundo\r"
        result = clean_text(text)
        assert '\r' not in result
        assert 'Hola' in result and 'Mundo' in result

    def test_normalizes_whitespace(self):
        """Should normalize multiple spaces/tabs to single space"""
        text = "Hola    \t  mundo"
        result = clean_text(text)
        assert result == "Hola mundo"

    def test_removes_excessive_newlines(self):
        """Should reduce 3+ newlines to 2"""
        text = "Párrafo 1\n\n\n\nPárrafo 2"
        result = clean_text(text)
        assert result == "Párrafo 1\n\nPárrafo 2"

    def test_removes_footer_patterns(self):
        """Should remove confidentiality footers"""
        text = "Contenido importante\n\nAviso de confidencialidad: Este correo es privado"
        result = clean_text(text)
        assert "Aviso de confidencialidad" not in result
        assert "Contenido importante" in result

    def test_strips_whitespace(self):
        """Should strip leading/trailing whitespace"""
        text = "   Texto con espacios   "
        result = clean_text(text)
        assert result == "Texto con espacios"

    def test_empty_string(self):
        """Should handle empty string"""
        result = clean_text("")
        assert result == ""

    def test_preserves_single_newlines(self):
        """Should preserve single newlines"""
        text = "Línea 1\nLínea 2"
        result = clean_text(text)
        assert result == "Línea 1\nLínea 2"

    def test_handles_spanish_characters(self):
        """Should preserve Spanish accents and ñ"""
        text = "José coordina la reunión mañana"
        result = clean_text(text)
        assert result == "José coordina la reunión mañana"

    def test_complex_case(self):
        """Should handle complex real-world case"""
        text = """
        Juan debe enviar   el    reporte.


        María   coordinará la reunión.

        Aviso de confidencialidad: privado
        """
        result = clean_text(text)
        assert "Juan debe enviar el reporte" in result
        assert "María coordinará la reunión" in result
        assert "Aviso de confidencialidad" not in result
        assert result.count('\n\n') <= 1  # At most one double newline
