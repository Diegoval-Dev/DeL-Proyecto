import re

# Abreviaturas comunes que NO deben cortar oración
ABBRS = {"Sr.", "Sra.", "Srta.", "Dr.", "Dra.", "Ing.", "Lic.", "Arq.", "Av.", "No.", "Nº.", "p.ej.", "etc."}

def split_sentences(text: str):
    """
    Divide texto en oraciones, evitando dividir en abreviaturas comunes.
    """
    # Dividir por punto seguido de espacio
    parts = re.split(r'(?<=[.!?])\s+', text)

    # Reunir partes que terminan en abreviatura
    sentences = []
    current = []

    for part in parts:
        current.append(part)
        joined = " ".join(current)

        # Verificar si termina en abreviatura
        ends_with_abbr = any(joined.endswith(abbr) for abbr in ABBRS)

        if not ends_with_abbr:
            sentences.append(joined.strip())
            current = []

    # Agregar lo que quede
    if current:
        sentences.append(" ".join(current).strip())

    # Filtrar vacíos
    sentences = [s for s in sentences if s]

    return sentences
