import regex as re
from datetime import datetime
import dateparser

ABS1 = re.compile(r"\b(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b", re.IGNORECASE)
ABS2 = re.compile(r"\b(\d{1,2}\s+de\s+[a-záéíóú]+(?:\s+de\s+\d{4})?)\b", re.IGNORECASE)
REL  = re.compile(r"\b(hoy|mañana|pasado mañana|esta semana|próxima semana|el lunes|el martes|el miércoles|el jueves|el viernes|el sábado|el domingo)\b", re.IGNORECASE)

def _parse_first(match_text: str, base_date: str):
    base = datetime.fromisoformat(base_date)
    dt = dateparser.parse(match_text, languages=["es"], settings={"RELATIVE_BASE": base})
    return dt

def extract_date_iso(sentence: str, base_date: str) -> str:
    for rx in (ABS1, ABS2, REL):
        m = rx.search(sentence)
        if m:
            dt = _parse_first(m.group(0), base_date)
            if dt:
                return dt.date().isoformat()
    return ""
