import re

# Abreviaturas comunes que NO deben cortar oración
ABBR = r"(Sr|Sra|Srta|Dr|Dra|Ing|Lic|Arq|Av|No|Nº|p\.ej|etc)\."

_SPLIT = re.compile(
    rf"""
    (?<!{ABBR})      # no dividir después de abreviatura
    (?<=[\.\?\!])    # punto final real
    \s+              # espacio
    """,
    re.IGNORECASE | re.VERBOSE
)

def split_sentences(text: str):
    parts = _SPLIT.split(text)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts
