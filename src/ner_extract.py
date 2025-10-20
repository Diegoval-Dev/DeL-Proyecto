from transformers import pipeline
import regex as re

_ACTION_VERBS = [
    "enviar","preparar","entregar","revisar","coordinar","agendar","subir",
    "compartir","firmar","actualizar","configurar","documentar","notificar",
    "resolver","investigar","programar","instalar","comprar","validar",
    "corregir","reportar"
]

_ner = None
def _get_ner():
    global _ner
    if _ner is None:
        _ner = pipeline(
            "token-classification",
            model="mrm8488/bert-spanish-cased-finetuned-ner",
            aggregation_strategy="simple"
        )
    return _ner

def _find_action_idx(tokens_lower):
    indices = [i for i,t in enumerate(tokens_lower) if t in _ACTION_VERBS]
    if not indices:
        return None
    mid = len(tokens_lower)//2
    return min(indices, key=lambda i: abs(i-mid))

def extract_person_responsable(sentence: str) -> str:
    ner = _get_ner()
    ents = ner(sentence)
    persons = [e for e in ents if e.get("entity_group","") == "PER"]
    if not persons:
        return "pendiente de asignar"

    toks = sentence.split()
    toks_lower = [t.lower().strip(".,;:()") for t in toks]
    verb_idx = _find_action_idx(toks_lower)

    if verb_idx is None:
        return persons[0]["word"].replace(" ##", "")

    best = None
    best_dist = 10**9
    for p in persons:
        span = p["word"].replace(" ##", "")
        try:
            idx = toks_lower.index(span.split()[0].lower())
        except ValueError:
            continue
        dist = abs(idx - verb_idx) - (0.1 if idx < verb_idx else 0.0)
        if dist < best_dist:
            best = span
            best_dist = dist

    return best if best else persons[0]["word"].replace(" ##", "")
