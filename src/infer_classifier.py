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
        self._bert_model = None
        self._bert_tokenizer = None
        self._model_type = None  # 'embeddings' o 'bert'
        self._th = 0.5
        self._load_if_available()

    def _load_if_available(self):
        clf_p = self.model_dir / "classifier.pkl"
        # Buscar encoder.pkl primero, luego sentence_encoder.pkl (compatibilidad)
        enc_p = self.model_dir / "encoder.pkl"
        if not enc_p.exists():
            enc_p = self.model_dir / "sentence_encoder.pkl"

        th_p  = self.model_dir / "threshold.txt"

        # Intentar cargar modelo de embeddings + LogReg
        if clf_p.exists() and enc_p.exists():
            self._clf = joblib.load(clf_p)
            self._enc = joblib.load(enc_p)
            self._model_type = 'embeddings'
            print(f"✓ Modelo Embeddings cargado desde: {self.model_dir}")
            if th_p.exists():
                try:
                    self._th = float(th_p.read_text().strip())
                    print(f"✓ Umbral de decisión: {self._th}")
                except Exception:
                    self._th = 0.5
        # Intentar cargar modelo BERT
        elif (self.model_dir / "config.json").exists():
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch

                self._bert_tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
                self._bert_model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
                self._bert_model.eval()
                self._model_type = 'bert'
                print(f"✓ Modelo BERT cargado desde: {self.model_dir}")
            except Exception as e:
                print(f"⚠️ Error cargando modelo BERT: {e}")
                print(f"⚠️ Usando reglas heurísticas")
        else:
            print(f"⚠️ No se encontró modelo en {self.model_dir}, usando reglas heurísticas")

    def _rules_score(self, s: str) -> float:
        t = s.lower()
        verb_hit = any(v in t for v in _ACTION_VERBS)
        must_hit = any(p in t for p in _OBLIGATION)
        return 0.9 if (verb_hit or must_hit) else 0.1

    def predict_sentence(self, s: str) -> Tuple[bool, float]:
        # Modelo de embeddings + LogReg
        if self._model_type == 'embeddings' and self._clf is not None and self._enc is not None:
            emb = self._enc.encode([s], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            score = float(self._clf.predict_proba(emb)[0,1])
            return (score >= self._th, score)

        # Modelo BERT
        elif self._model_type == 'bert' and self._bert_model is not None:
            import torch

            inputs = self._bert_tokenizer(
                s,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )

            with torch.no_grad():
                outputs = self._bert_model(**inputs)
                logits = outputs.logits
                probas = torch.softmax(logits, dim=-1).numpy()[0]

            score = float(probas[1])  # Probabilidad de clase TAREA
            return (score >= self._th, score)

        # Fallback por reglas
        score = self._rules_score(s)
        return (score >= 0.5, score)
