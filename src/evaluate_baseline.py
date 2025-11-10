#!/usr/bin/env python3
"""
Evaluación del mejor modelo baseline en test set
"""

import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main():
    base_dir = Path(__file__).parent.parent
    model_dir = base_dir / "models" / "best_baseline"
    test_path = base_dir / "data" / "splits" / "test.jsonl"
    output_dir = base_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("EVALUACIÓN DEL MODELO BASELINE EN TEST SET")
    print("="*80)

    # Cargar modelo
    print(f"\nCargando modelo desde {model_dir}...")
    encoder = joblib.load(model_dir / "encoder.pkl")
    classifier = joblib.load(model_dir / "classifier.pkl")
    print("✓ Modelo cargado")

    # Cargar test data
    print(f"\nCargando test set...")
    test_data = load_jsonl(test_path)
    X_test_text = [d['text'] for d in test_data]
    y_test = np.array([1 if d['label'] == 'TAREA' else 0 for d in test_data])
    print(f"Test set: {len(X_test_text)} oraciones")

    # Generar embeddings
    print("\nGenerando embeddings...")
    X_test = encoder.encode(
        X_test_text,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # Predicciones
    print("\nRealizando predicciones...")
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)

    # Métricas
    f1 = f1_score(y_test, y_pred)

    print("\n" + "="*80)
    print("RESULTADOS EN TEST SET")
    print("="*80)
    print(f"\nF1 Score: {f1:.4f}")

    print("\nReporte de clasificación:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=['NO_TAREA', 'TAREA'],
        digits=4
    ))

    print("Matriz de confusión:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Guardar resultados
    results = {
        'model': 'best_baseline',
        'test_size': len(y_test),
        'f1': float(f1),
        'classification_report': classification_report(
            y_test, y_pred, target_names=['NO_TAREA', 'TAREA'], output_dict=True
        ),
        'confusion_matrix': cm.tolist()
    }

    output_path = output_dir / "baseline_test_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Resultados guardados en {output_path}")
    print("="*80)

if __name__ == "__main__":
    main()
