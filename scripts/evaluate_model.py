"""
Script simple para evaluar el modelo re-entrenado en el test set.
"""
import json
import sys
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from infer_classifier import SentenceTaskClassifier


def main():
    base_dir = Path(__file__).parent.parent
    test_file = base_dir / "data" / "splits" / "test.jsonl"
    model_dir = base_dir / "models" / "best_baseline"

    # Cargar test data
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))

    texts = [d['text'] for d in test_data]
    y_true = [1 if d['label'] == 'TAREA' else 0 for d in test_data]

    print("="*80)
    print("EVALUACIÓN EN TEST SET")
    print("="*80)
    print()
    print(f"Test set: {len(texts)} oraciones")
    print(f"  TAREA: {sum(y_true)} ({100*sum(y_true)/len(y_true):.1f}%)")
    print(f"  NO_TAREA: {len(y_true)-sum(y_true)} ({100*(len(y_true)-sum(y_true))/len(y_true):.1f}%)")
    print()

    # Cargar modelo y predecir
    print(f"Cargando modelo desde: {model_dir}")
    clf = SentenceTaskClassifier(model_dir=model_dir)
    print("✓ Modelo cargado")
    print()

    print("Prediciendo...")
    y_pred = []
    scores = []
    for text in texts:
        is_task, score = clf.predict_sentence(text)
        y_pred.append(1 if is_task else 0)
        scores.append(score)

    print("✓ Predicciones completadas")
    print()

    # Calcular métricas
    print("="*80)
    print("MÉTRICAS DE CLASIFICACIÓN")
    print("="*80)
    print()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1
    )

    print(f"F1 Score (TAREA): {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print()

    print("Reporte detallado:")
    print(classification_report(y_true, y_pred, target_names=['NO_TAREA', 'TAREA'], digits=4))
    print()

    print("Matriz de confusión:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print()
    print("[[TN, FP],")
    print(" [FN, TP]]")
    print()

    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    print()

    # Guardar resultados
    results = {
        "test_size": len(texts),
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }

    output_path = base_dir / "eval" / "test_results_v2.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"📄 Resultados guardados en: {output_path}")
    print()

    # Comparar con versión anterior
    v1_path = base_dir / "eval" / "baseline_test_results.json"
    if v1_path.exists():
        with open(v1_path, 'r', encoding='utf-8') as f:
            v1_results = json.load(f)

        print("="*80)
        print("COMPARACIÓN CON MODELO ANTERIOR")
        print("="*80)
        print()
        print(f"{'Métrica':<20} {'V1 (Original)':<20} {'V2 (Re-entrenado)':<20} {'Cambio'}")
        print("-"*80)
        print(f"{'F1 Score':<20} {v1_results.get('f1_task', 0):<20.4f} {f1:<20.4f} {f1-v1_results.get('f1_task', 0):+.4f}")
        print(f"{'Precision':<20} {v1_results.get('precision_task', 0):<20.4f} {precision:<20.4f} {precision-v1_results.get('precision_task', 0):+.4f}")
        print(f"{'Recall':<20} {v1_results.get('recall_task', 0):<20.4f} {recall:<20.4f} {recall-v1_results.get('recall_task', 0):+.4f}")
        print(f"{'Falsos Positivos':<20} {v1_results.get('confusion_matrix', [[0,0],[0,0]])[0][1]:<20} {fp:<20} {fp-v1_results.get('confusion_matrix', [[0,0],[0,0]])[0][1]:+}")
        print()


if __name__ == "__main__":
    main()
