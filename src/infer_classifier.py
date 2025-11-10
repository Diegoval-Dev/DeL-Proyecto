import joblib, os
import numpy as np
from pathlib import Path
from typing import Tuple

_ACTION_VERBS = {
    "enviar","preparar","entregar","revisar","coordinar","agendar","subir",
    "compartir","firmar","actualizar","configurar","documentar","notificar",
    "resolver","investigar","programar","instalar","comprar","validar",
    "corregir","reportar"
}
_OBLIGATION = {"debe","deben","tiene que","tienen que","hay que","se acuerda que","se decidió que"}

class SentenceTaskClassifier:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self._clf = None
        self._enc = None
        self._th = 0.5
        self._load_if_available()

    def _load_if_available(self):
        clf_p = self.model_dir / "classifier.pkl"
        # Buscar encoder.pkl primero, luego sentence_encoder.pkl (compatibilidad)
        enc_p = self.model_dir / "encoder.pkl"
        if not enc_p.exists():
            enc_p = self.model_dir / "sentence_encoder.pkl"

        th_p  = self.model_dir / "threshold.txt"

        if clf_p.exists() and enc_p.exists():
            self._clf = joblib.load(clf_p)
            self._enc = joblib.load(enc_p)
            print(f"✓ Modelo cargado desde: {self.model_dir}")
            if th_p.exists():
                try:
                    self._th = float(th_p.read_text().strip())
                    print(f"✓ Umbral de decisión: {self._th}")
                except Exception:
                    self._th = 0.5
        else:
            print(f"⚠️ No se encontró modelo en {self.model_dir}, usando reglas heurísticas")

    def _rules_score(self, s: str) -> float:
        t = s.lower()
        verb_hit = any(v in t for v in _ACTION_VERBS)
        must_hit = any(p in t for p in _OBLIGATION)
        return 0.9 if (verb_hit or must_hit) else 0.1

    def predict_sentence(self, s: str) -> Tuple[bool, float]:
        if self._clf is not None and self._enc is not None:
            emb = self._enc.encode([s], convert_to_numpy=True, normalize_embeddings=True)
            score = float(self._clf.predict_proba(emb)[0,1])
            return (score >= self._th, score)
        # Fallback por reglas
        score = self._rules_score(s)
        return (score >= 0.5, score)
