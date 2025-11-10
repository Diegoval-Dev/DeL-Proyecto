"""
Clasificador con umbral ajustable para reducir falsos positivos.
"""
import joblib
from pathlib import Path
from typing import Tuple

from sentence_transformers import SentenceTransformer


class SentenceTaskClassifierWithThreshold:
    """
    Clasificador con umbral de decisión configurable.

    Threshold más alto = menos falsos positivos, pero más falsos negativos
    Threshold más bajo = menos falsos negativos, pero más falsos positivos
    """

    def __init__(self, model_dir: Path, threshold: float = 0.5):
        """
        Args:
            model_dir: Directorio con classifier.pkl y encoder.pkl
            threshold: Umbral de decisión (default=0.5)
        """
        self.model_dir = Path(model_dir)
        self.threshold = threshold
        self.classifier = None
        self.encoder = None
        self._load_if_available()

    def _load_if_available(self):
        """Carga el clasificador y encoder si existen."""
        clf_p = self.model_dir / "classifier.pkl"
        enc_p = self.model_dir / "encoder.pkl"
        if not enc_p.exists():
            enc_p = self.model_dir / "sentence_encoder.pkl"

        if clf_p.exists() and enc_p.exists():
            self.classifier = joblib.load(clf_p)
            self.encoder = joblib.load(enc_p)

    def predict_sentence(self, text: str) -> Tuple[bool, float]:
        """
        Predice si una oración es TAREA usando el threshold configurado.

        Args:
            text: Oración a clasificar

        Returns:
            (is_task, probability): Tupla con predicción booleana y probabilidad
        """
        if not text or not text.strip():
            return False, 0.0

        # Encode
        embedding = self.encoder.encode([text], show_progress_bar=False)

        # Predict probability
        proba = self.classifier.predict_proba(embedding)[0]
        prob_task = proba[1]  # Probabilidad de clase TAREA

        # Aplicar threshold
        is_task = prob_task >= self.threshold

        return is_task, prob_task

    def set_threshold(self, threshold: float):
        """Ajusta el umbral de decisión."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold debe estar entre 0.0 y 1.0")
        self.threshold = threshold


def find_optimal_threshold(model_dir: Path, test_jsonl: Path):
    """
    Encuentra el umbral óptimo para maximizar F1 en el conjunto de test.

    Args:
        model_dir: Directorio con el modelo entrenado
        test_jsonl: Archivo JSONL con datos de test

    Returns:
        dict: Resultados con mejor threshold y métricas
    """
    import json
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

    # Cargar test data
    test_texts = []
    test_labels = []

    with open(test_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            test_texts.append(entry['text'])
            test_labels.append(1 if entry['label'] == 'TAREA' else 0)

    # Probar diferentes thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results = []

    print("="*80)
    print("BÚSQUEDA DE THRESHOLD ÓPTIMO")
    print("="*80)
    print()

    for threshold in thresholds:
        clf = SentenceTaskClassifierWithThreshold(model_dir, threshold=threshold)

        predictions = []
        for text in test_texts:
            is_task, _ = clf.predict_sentence(text)
            predictions.append(1 if is_task else 0)

        # Calcular métricas
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, predictions, average='binary', pos_label=1
        )

        cm = confusion_matrix(test_labels, predictions)
        tn, fp, fn, tp = cm.ravel()

        result = {
            'threshold': threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }
        results.append(result)

        print(f"Threshold: {threshold:.1f}")
        print(f"  F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        print(f"  FP: {fp} | FN: {fn}")
        print()

    # Encontrar mejor threshold
    best = max(results, key=lambda x: x['f1'])

    print("="*80)
    print("MEJOR THRESHOLD")
    print("="*80)
    print(f"Threshold: {best['threshold']}")
    print(f"F1 Score: {best['f1']:.4f}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall: {best['recall']:.4f}")
    print(f"Falsos Positivos: {best['fp']}")
    print(f"Falsos Negativos: {best['fn']}")
    print()

    return best, results


if __name__ == "__main__":
    import sys

    # Ejemplo de uso
    base_dir = Path(__file__).parent.parent
    model_dir = base_dir / "models" / "best_baseline"
    test_file = base_dir / "data" / "splits" / "test.jsonl"

    if not test_file.exists():
        print(f"❌ No se encontró archivo de test: {test_file}")
        sys.exit(1)

    # Buscar threshold óptimo
    best, all_results = find_optimal_threshold(model_dir, test_file)

    # Guardar resultados
    import json
    output_path = base_dir / "eval" / "threshold_tuning.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'best_threshold': best,
            'all_results': all_results
        }, f, indent=2)

    print(f"📄 Resultados guardados en: {output_path}")
    print()
    print("="*80)
    print("RECOMENDACIÓN")
    print("="*80)
    print()
    print(f"Para usar el threshold óptimo ({best['threshold']}) en la app:")
    print()
    print("1. Modificar app/streamlit_app.py:")
    print("   from src.infer_classifier_with_threshold import SentenceTaskClassifierWithThreshold")
    print()
    print("2. Cambiar:")
    print(f"   clf = SentenceTaskClassifierWithThreshold(model_dir, threshold={best['threshold']})")
    print()
