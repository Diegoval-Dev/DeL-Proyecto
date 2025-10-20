import re

_FOOTER_PAT = re.compile(r"(Aviso de confidencialidad.*|Este correo.*)$", re.IGNORECASE | re.MULTILINE)

def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _FOOTER_PAT.sub("", text)
    return text.strip()
