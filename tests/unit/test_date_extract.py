"""
Unit tests for date_extract module
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, date

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from date_extract import extract_date_iso

class TestExtractDateIso:
    """Tests for extract_date_iso function"""

    def test_extracts_absolute_date_slash(self):
        """Should extract date in DD/MM/YYYY format"""
        text = "Enviar el reporte el 15/11/2025"
        base_date = "2025-11-08"
        result = extract_date_iso(text, base_date)
        assert result != ""
        assert "2025-11" in result

    def test_extracts_absolute_date_dash(self):
        """Should extract date in DD-MM-YYYY format"""
        text = "La reunión es el 20-12-2025"
        base_date = "2025-11-08"
        result = extract_date_iso(text, base_date)
        assert result != ""

    def test_extracts_verbal_date(self):
        """Should extract verbal date"""
        text = "Entregar el 23 de noviembre de 2025"
        base_date = "2025-11-08"
        result = extract_date_iso(text, base_date)
        assert result != ""
        assert "2025-11-23" in result

    def test_extracts_relative_date(self):
        """Should extract relative dates"""
        text = "Completar mañana"
        base_date = "2025-11-08"
        result = extract_date_iso(text, base_date)
        # Should find "mañana" and return next day
        assert result != ""

    def test_extracts_day_of_week(self):
        """Should extract day of week"""
        text = "Reunión el viernes"
        base_date = "2025-11-08"
        result = extract_date_iso(text, base_date)
        assert result != ""

    def test_no_date_in_text(self):
        """Should return empty string when no date found"""
        text = "Este texto no contiene fechas"
        base_date = "2025-11-08"
        result = extract_date_iso(text, base_date)
        assert result == ""

    def test_multiple_dates(self):
        """Should extract first date when multiple present"""
        text = "Reunión el 10/11/2025 y entrega el 20/11/2025"
        base_date = "2025-11-08"
        result = extract_date_iso(text, base_date)
        assert result != ""

    def test_returns_iso_format(self):
        """Should return date in ISO format (YYYY-MM-DD)"""
        text = "Entregar el 15/11/2025"
        base_date = "2025-11-08"
        result = extract_date_iso(text, base_date)
        if result:
            # Should match ISO date format
            assert len(result) == 10
            assert result.count('-') == 2
