"""
Script mejorado para ajustar el umbral de clasificación
Objetivo: Minimizar falsos positivos manteniendo recall alto
"""
import json
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from infer_classifier import SentenceTaskClassifier

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def evaluate_threshold(clf, texts, y_true, threshold):
    """Evalua modelo con un umbral específico"""
    y_pred = []
    scores = []

    for text in texts:
        _, score = clf.predict_sentence(text)
        y_pred.append(1 if score >= threshold else 0)
        scores.append(score)

    # Calcular métricas
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)

    # Calcular falsos positivos y negativos
    fp = sum((pred == 1 and true == 0) for pred, true in zip(y_pred, y_true))
    fn = sum((pred == 0 and true == 1) for pred, true in zip(y_pred, y_true))
    tp = sum((pred == 1 and true == 1) for pred, true in zip(y_pred, y_true))
    tn = sum((pred == 0 and true == 0) for pred, true in zip(y_pred, y_true))

    return {
        "threshold": threshold,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tn": tn
    }

def main():
    # Cargar dev set
    dev_path = Path("data/splits/dev.jsonl")
    data = list(load_jsonl(dev_path))
    texts = [d["text"] for d in data]
    y_true = [1 if d["label"] == "TAREA" else 0 for d in data]

    print("=" * 80)
    print("AJUSTE DE UMBRAL - REDUCCIÓN DE FALSOS POSITIVOS")
    print("=" * 80)
    print(f"\nDataset: {len(texts)} oraciones")
    print(f"Balance: {sum(y_true)} TAREA / {len(y_true) - sum(y_true)} NO_TAREA")
    print()

    # Cargar modelo
    clf = SentenceTaskClassifier(Path("models/best_baseline"))

    # Probar diferentes umbrales
    thresholds = np.arange(0.3, 0.9, 0.05)
    results = []

    print("Evaluando umbrales...")
    print()
    print(f"{'Threshold':<10} {'F1':<8} {'Precision':<10} {'Recall':<8} {'FP':<5} {'FN':<5}")
    print("-" * 60)

    for threshold in thresholds:
        result = evaluate_threshold(clf, texts, y_true, threshold)
        results.append(result)

        print(f"{result['threshold']:.2f}       "
              f"{result['f1']:.4f}   "
              f"{result['precision']:.4f}     "
              f"{result['recall']:.4f}   "
              f"{result['fp']:<5}  "
              f"{result['fn']:<5}")

    # Encontrar mejor umbral
    print()
    print("=" * 80)
    print("ANÁLISIS DE RESULTADOS")
    print("=" * 80)
    print()

    # Mejor F1
    best_f1 = max(results, key=lambda x: x['f1'])
    print(f"Mejor F1: {best_f1['f1']:.4f} (umbral={best_f1['threshold']:.2f})")

    # Mejor precision (minimiza FP)
    best_precision = max(results, key=lambda x: x['precision'])
    print(f"Mejor Precision: {best_precision['precision']:.4f} (umbral={best_precision['threshold']:.2f}, FP={best_precision['fp']})")

    # Mejor recall (minimiza FN)
    best_recall = max(results, key=lambda x: x['recall'])
    print(f"Mejor Recall: {best_recall['recall']:.4f} (umbral={best_recall['threshold']:.2f}, FN={best_recall['fn']})")

    # Mejor balance: F1 alto con mínimos FP
    # Filtrar solo los que tienen F1 > 0.95
    high_f1_results = [r for r in results if r['f1'] >= 0.95]
    if high_f1_results:
        best_balance = min(high_f1_results, key=lambda x: x['fp'])
        print(f"\nMejor balance (F1≥0.95 con mínimos FP):")
        print(f"  Umbral: {best_balance['threshold']:.2f}")
        print(f"  F1: {best_balance['f1']:.4f}")
        print(f"  Precision: {best_balance['precision']:.4f}")
        print(f"  Recall: {best_balance['recall']:.4f}")
        print(f"  Falsos Positivos: {best_balance['fp']}")
        print(f"  Falsos Negativos: {best_balance['fn']}")
    else:
        best_balance = best_f1
        print(f"\nNo hay umbrales con F1≥0.95, usando mejor F1")

    print()
    print("=" * 80)
    print("RECOMENDACIÓN")
    print("=" * 80)

    # Recomendar umbral
    if best_balance['fp'] <= 2:
        recommended = best_balance
        print(f"✓ Umbral recomendado: {recommended['threshold']:.2f}")
        print(f"  Este umbral minimiza falsos positivos ({recommended['fp']}) manteniendo F1 alto ({recommended['f1']:.4f})")
    else:
        # Buscar umbral que tenga FP <= 2
        low_fp_results = [r for r in results if r['fp'] <= 2]
        if low_fp_results:
            recommended = max(low_fp_results, key=lambda x: x['f1'])
            print(f"✓ Umbral recomendado: {recommended['threshold']:.2f}")
            print(f"  Este umbral reduce falsos positivos a {recommended['fp']} con F1 de {recommended['f1']:.4f}")
        else:
            recommended = best_f1
            print(f"⚠ No se puede reducir FP a ≤2 sin sacrificar mucho F1")
            print(f"  Umbral recomendado: {recommended['threshold']:.2f} (mejor F1)")

    # Guardar umbral recomendado
    threshold_file = Path("models/best_baseline/threshold.txt")
    threshold_file.write_text(f"{recommended['threshold']:.2f}")
    print(f"\n✓ Umbral guardado en: {threshold_file}")

    # Guardar resultados completos
    results_file = Path("eval/threshold_tuning_improved.json")
    results_file.parent.mkdir(exist_ok=True, parents=True)

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "recommended_threshold": recommended,
            "all_results": results
        }, f, indent=2)

    print(f"✓ Resultados completos guardados en: {results_file}")

if __name__ == "__main__":
    main()
