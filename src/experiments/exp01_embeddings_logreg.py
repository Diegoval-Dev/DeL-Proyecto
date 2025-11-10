#!/usr/bin/env python3
"""
Experimento 1: Embeddings + Logistic Regression
Prueba múltiples modelos de sentence embeddings con grid search de LogisticRegression.
Target: F1 >= 0.75
"""

import json
import time
import joblib
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

def load_jsonl(path):
    """Carga dataset desde archivo JSONL"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def prepare_data(data):
    """Prepara textos y labels"""
    X_text = [d['text'] for d in data]
    y = np.array([1 if d['label'] == 'TAREA' else 0 for d in data])
    return X_text, y

# Modelos de embeddings a probar
EMBEDDING_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "hiiamsid/sentence_similarity_spanish_es"
]

# Grid de hiperparámetros para LogisticRegression
PARAM_GRID = {
    'C': [0.01, 0.1, 1.0, 10.0],
    'class_weight': ['balanced', None],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [2000]
}

def evaluate_model(clf, X, y, y_pred=None):
    """Calcula métricas de evaluación"""
    if y_pred is None:
        y_pred = clf.predict(X)

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0)
    }

    return metrics

def train_and_evaluate_embedding(
    embedding_model_name,
    X_train_text,
    y_train,
    X_dev_text,
    y_dev,
    param_grid
):
    """Entrena y evalúa un modelo de embeddings con grid search"""
    print(f"\n{'='*80}")
    print(f"Probando: {embedding_model_name}")
    print(f"{'='*80}")

    # Cargar encoder
    print("Cargando encoder...")
    start_time = time.time()
    encoder = SentenceTransformer(embedding_model_name)
    load_time = time.time() - start_time
    print(f"Encoder cargado en {load_time:.2f}s")

    # Generar embeddings
    print("Generando embeddings de entrenamiento...")
    start_time = time.time()
    X_train = encoder.encode(
        X_train_text,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    train_embed_time = time.time() - start_time
    print(f"Embeddings de train generados en {train_embed_time:.2f}s")
    print(f"Shape: {X_train.shape}")

    print("Generando embeddings de desarrollo...")
    start_time = time.time()
    X_dev = encoder.encode(
        X_dev_text,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    dev_embed_time = time.time() - start_time
    print(f"Embeddings de dev generados en {dev_embed_time:.2f}s")

    # Grid Search
    print("\nEjecutando Grid Search...")
    print(f"Parámetros a probar: {param_grid}")

    clf = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(
        clf,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"\nGrid Search completado en {train_time:.2f}s")
    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Mejor F1 (CV): {grid_search.best_score_:.4f}")

    # Evaluar en dev
    best_clf = grid_search.best_estimator_
    y_dev_pred = best_clf.predict(X_dev)

    dev_metrics = evaluate_model(best_clf, X_dev, y_dev, y_dev_pred)

    print(f"\nMétricas en DEV:")
    print(f"  Accuracy: {dev_metrics['accuracy']:.4f}")
    print(f"  Precision: {dev_metrics['precision']:.4f}")
    print(f"  Recall: {dev_metrics['recall']:.4f}")
    print(f"  F1: {dev_metrics['f1']:.4f}")

    print(f"\nReporte de clasificación:")
    print(classification_report(
        y_dev,
        y_dev_pred,
        target_names=['NO_TAREA', 'TAREA'],
        digits=4
    ))

    print(f"\nMatriz de confusión:")
    cm = confusion_matrix(y_dev, y_dev_pred)
    print(cm)

    # Calcular latencia promedio
    start_time = time.time()
    _ = best_clf.predict(X_dev[:100] if len(X_dev) >= 100 else X_dev)
    latency_total = time.time() - start_time
    latency_per_sentence = (latency_total / min(100, len(X_dev))) * 1000  # en ms

    results = {
        'model_name': embedding_model_name,
        'best_params': grid_search.best_params_,
        'cv_f1': float(grid_search.best_score_),
        'dev_metrics': {k: float(v) for k, v in dev_metrics.items()},
        'confusion_matrix': cm.tolist(),
        'timings': {
            'load_time': load_time,
            'train_embed_time': train_embed_time,
            'dev_embed_time': dev_embed_time,
            'train_time': train_time,
            'latency_ms_per_sentence': latency_per_sentence
        }
    }

    return encoder, best_clf, results

def main():
    # Rutas
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "data" / "splits"
    output_dir = base_dir / "models" / "exp01_embeddings_logreg"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("EXPERIMENTO 1: EMBEDDINGS + LOGISTIC REGRESSION")
    print("="*80)

    # Cargar datos
    print("\nCargando datasets...")
    train_data = load_jsonl(data_dir / "train.jsonl")
    dev_data = load_jsonl(data_dir / "dev.jsonl")

    X_train_text, y_train = prepare_data(train_data)
    X_dev_text, y_dev = prepare_data(dev_data)

    print(f"Train: {len(X_train_text)} oraciones")
    print(f"Dev: {len(X_dev_text)} oraciones")
    print(f"Balance train - TAREA: {100*y_train.sum()/len(y_train):.1f}%")
    print(f"Balance dev - TAREA: {100*y_dev.sum()/len(y_dev):.1f}%")

    # Probar cada modelo de embeddings
    all_results = []

    for embedding_model in EMBEDDING_MODELS:
        try:
            encoder, clf, results = train_and_evaluate_embedding(
                embedding_model,
                X_train_text,
                y_train,
                X_dev_text,
                y_dev,
                PARAM_GRID
            )

            all_results.append(results)

            # Guardar modelo
            model_name_safe = embedding_model.replace('/', '_')
            model_output_dir = output_dir / model_name_safe
            model_output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nGuardando modelo en {model_output_dir}...")
            joblib.dump(clf, model_output_dir / "classifier.pkl")
            joblib.dump(encoder, model_output_dir / "encoder.pkl")

            with open(model_output_dir / "results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print("✓ Modelo guardado")

        except Exception as e:
            print(f"\n✗ Error con {embedding_model}: {e}")
            continue

    # Comparar resultados y seleccionar mejor modelo
    print("\n" + "="*80)
    print("RESUMEN DE RESULTADOS")
    print("="*80)

    best_model = None
    best_f1 = 0

    for i, result in enumerate(all_results, 1):
        f1 = result['dev_metrics']['f1']
        print(f"\n{i}. {result['model_name']}")
        print(f"   F1 (dev): {f1:.4f}")
        print(f"   Accuracy: {result['dev_metrics']['accuracy']:.4f}")
        print(f"   Latencia: {result['timings']['latency_ms_per_sentence']:.2f}ms/oración")
        print(f"   Mejores params: {result['best_params']}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = result

    # Guardar resumen
    summary = {
        'experiment': 'exp01_embeddings_logreg',
        'models_tested': len(all_results),
        'all_results': all_results,
        'best_model': best_model['model_name'] if best_model else None,
        'best_f1': float(best_f1)
    }

    with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Mostrar mejor modelo
    if best_model:
        print("\n" + "="*80)
        print("MEJOR MODELO")
        print("="*80)
        print(f"Modelo: {best_model['model_name']}")
        print(f"F1: {best_f1:.4f}")
        print(f"Target alcanzado: {'✓ SÍ' if best_f1 >= 0.75 else '✗ NO'} (target: 0.75)")

        # Copiar mejor modelo al directorio raíz de models
        best_model_safe = best_model['model_name'].replace('/', '_')
        best_model_dir = output_dir / best_model_safe

        import shutil
        print(f"\nCopiando mejor modelo a models/best_baseline...")
        best_dir = base_dir / "models" / "best_baseline"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(best_model_dir, best_dir)
        print("✓ Mejor modelo copiado")

    print("\n" + "="*80)
    print("✓ EXPERIMENTO 1 COMPLETADO")
    print("="*80)

if __name__ == "__main__":
    main()
