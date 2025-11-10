#!/usr/bin/env python3
"""
Experimento 3 MEJORADO: Weighted Ensemble
Aprende pesos óptimos para cada modelo usando optimización
"""

import json
import joblib
import numpy as np
from pathlib import Path
import torch
from scipy.optimize import minimize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def load_jsonl(path):
    """Carga dataset desde JSONL"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def predict_embeddings_model(model_dir, texts):
    """Predicciones con modelo de embeddings + LogReg"""
    encoder = joblib.load(model_dir / "encoder.pkl")
    classifier = joblib.load(model_dir / "classifier.pkl")

    X = encoder.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    probas = classifier.predict_proba(X)
    return probas

def predict_bert_model(model_dir, texts):
    """Predicciones con modelo BERT fine-tuned"""
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    probas_list = []

    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probas = torch.softmax(logits, dim=-1).numpy()

        probas_list.append(probas)

    all_probas = np.vstack(probas_list)
    return all_probas

def learn_optimal_weights(models_probas, y_true, method='f1'):
    """
    Aprende pesos óptimos para ensemble usando optimización

    Args:
        models_probas: Lista de arrays de probabilidades [n_models, n_samples, 2]
        y_true: Labels verdaderos
        method: Métrica a optimizar ('f1', 'accuracy', 'precision')

    Returns:
        Array con pesos óptimos (suman 1)
    """
    print(f"\n{'='*80}")
    print("OPTIMIZACIÓN DE PESOS DEL ENSEMBLE")
    print(f"{'='*80}")
    print(f"Número de modelos: {len(models_probas)}")
    print(f"Métrica a maximizar: {method}")

    def objective_function(weights):
        """Función objetivo: maximizar métrica"""
        # Normalizar pesos para que sumen 1
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Calcular ensemble probabilities
        ensemble_probas = np.zeros_like(models_probas[0])
        for i, probas in enumerate(models_probas):
            ensemble_probas += weights[i] * probas

        # Predicciones
        y_pred = np.argmax(ensemble_probas, axis=1)

        # Calcular métrica
        if method == 'f1':
            score = f1_score(y_true, y_pred)
        elif method == 'accuracy':
            score = (y_pred == y_true).mean()
        elif method == 'precision':
            from sklearn.metrics import precision_score
            score = precision_score(y_true, y_pred, zero_division=0)

        # Minimizamos el negativo (porque minimize busca mínimo)
        return -score

    # Inicializar con pesos uniformes
    n_models = len(models_probas)
    initial_weights = np.ones(n_models) / n_models

    # Restricciones: pesos entre 0 y 1
    bounds = [(0.0, 1.0)] * n_models

    # Restricción: suma = 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

    print("\nEjecutando optimización...")
    result = minimize(
        objective_function,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 100}
    )

    if result.success:
        optimal_weights = result.x / result.x.sum()  # Renormalizar
        optimal_score = -result.fun

        print(f"✓ Optimización exitosa")
        print(f"\nPesos iniciales (uniforme): {initial_weights}")
        print(f"Pesos optimizados:          {optimal_weights}")
        print(f"\n{method.upper()} optimizado: {optimal_score:.4f}")

        return optimal_weights
    else:
        print(f"✗ Optimización falló: {result.message}")
        return initial_weights

def weighted_ensemble_predict(models_probas, weights):
    """Predicción con ensemble pesado"""
    ensemble_probas = np.zeros_like(models_probas[0])

    for i, probas in enumerate(models_probas):
        ensemble_probas += weights[i] * probas

    predictions = np.argmax(ensemble_probas, axis=1)
    return predictions, ensemble_probas

def main():
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "data" / "splits"
    output_dir = base_dir / "models" / "exp03_ensemble_weighted"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("EXPERIMENTO 3: WEIGHTED ENSEMBLE")
    print("="*80)

    # Cargar dev data
    print("\nCargando dev set...")
    dev_data = load_jsonl(data_dir / "dev.jsonl")
    texts = [d['text'] for d in dev_data]
    y_true = np.array([1 if d['label'] == 'TAREA' else 0 for d in dev_data])

    print(f"Dev set: {len(texts)} oraciones")

    # Modelos a ensamblar
    models_config = [
        {
            'name': 'Embeddings (hiiamsid)',
            'type': 'embeddings',
            'path': base_dir / "models" / "best_baseline"
        },
        {
            'name': 'BERT español (dccuchile)',
            'type': 'bert',
            'path': base_dir / "models" / "exp02_bert_finetuning" / "dccuchile_bert-base-spanish-wwm-cased" / "best_model"
        },
        {
            'name': 'BERT multilingüe',
            'type': 'bert',
            'path': base_dir / "models" / "exp02_bert_finetuning" / "bert-base-multilingual-cased" / "best_model"
        }
    ]

    # Obtener predicciones de cada modelo
    all_probas = []
    individual_results = []

    for model_info in models_config:
        print(f"\n{'='*80}")
        print(f"Evaluando: {model_info['name']}")
        print(f"{'='*80}")

        try:
            if model_info['type'] == 'embeddings':
                probas = predict_embeddings_model(model_info['path'], texts)
            elif model_info['type'] == 'bert':
                probas = predict_bert_model(model_info['path'], texts)

            preds = np.argmax(probas, axis=1)
            f1 = f1_score(y_true, preds)

            print(f"F1 individual: {f1:.4f}")

            all_probas.append(probas)
            individual_results.append({
                'name': model_info['name'],
                'f1': float(f1)
            })

        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    if len(all_probas) < 2:
        print("\n✗ No hay suficientes modelos para ensemble")
        return

    # ========================================================================
    # ENSEMBLE CON PESOS UNIFORMES (baseline)
    # ========================================================================

    print(f"\n{'='*80}")
    print("ENSEMBLE CON PESOS UNIFORMES")
    print(f"{'='*80}")

    uniform_weights = np.ones(len(all_probas)) / len(all_probas)
    uniform_preds, _ = weighted_ensemble_predict(all_probas, uniform_weights)
    uniform_f1 = f1_score(y_true, uniform_preds)

    print(f"Pesos uniformes: {uniform_weights}")
    print(f"F1 Score: {uniform_f1:.4f}")

    # ========================================================================
    # ENSEMBLE CON PESOS OPTIMIZADOS
    # ========================================================================

    optimal_weights = learn_optimal_weights(all_probas, y_true, method='f1')

    optimal_preds, optimal_probas = weighted_ensemble_predict(all_probas, optimal_weights)
    optimal_f1 = f1_score(y_true, optimal_preds)

    print(f"\n{'='*80}")
    print("RESULTADOS FINALES")
    print(f"{'='*80}")

    print(f"\nF1 Score:")
    print(f"  Ensemble uniforme:   {uniform_f1:.4f}")
    print(f"  Ensemble optimizado: {optimal_f1:.4f}")
    print(f"  Mejora:              {optimal_f1 - uniform_f1:+.4f}")

    # Matriz de confusión
    cm = confusion_matrix(y_true, optimal_preds)
    print(f"\nMatriz de confusión (optimizado):")
    print(cm)

    print(f"\nReporte de clasificación:")
    print(classification_report(
        y_true,
        optimal_preds,
        target_names=['NO_TAREA', 'TAREA'],
        digits=4
    ))

    # ========================================================================
    # COMPARACIÓN CON MODELOS INDIVIDUALES
    # ========================================================================

    print(f"\n{'='*80}")
    print("COMPARACIÓN COMPLETA")
    print(f"{'='*80}")

    print("\nModelos individuales:")
    for res in individual_results:
        print(f"  {res['name']:<30}: F1 = {res['f1']:.4f}")

    print(f"\nEnsembles:")
    print(f"  {'Uniforme':<30}: F1 = {uniform_f1:.4f}")
    print(f"  {'Optimizado':<30}: F1 = {optimal_f1:.4f}")

    best_individual = max(individual_results, key=lambda x: x['f1'])
    improvement_over_best = optimal_f1 - best_individual['f1']

    print(f"\nMejora sobre mejor individual: {improvement_over_best:+.4f}")

    # ========================================================================
    # GUARDAR RESULTADOS
    # ========================================================================

    results = {
        'experiment': 'exp03_weighted_ensemble',
        'num_models': len(all_probas),
        'individual_models': individual_results,
        'uniform_ensemble': {
            'weights': uniform_weights.tolist(),
            'f1': float(uniform_f1)
        },
        'optimized_ensemble': {
            'weights': optimal_weights.tolist(),
            'f1': float(optimal_f1),
            'confusion_matrix': cm.tolist()
        },
        'improvement_over_uniform': float(optimal_f1 - uniform_f1),
        'improvement_over_best_individual': float(improvement_over_best)
    }

    output_file = output_dir / "results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Resultados guardados en: {output_file}")

    # Guardar modelo (pesos)
    weights_file = output_dir / "optimal_weights.json"
    with open(weights_file, 'w', encoding='utf-8') as f:
        json.dump({
            'weights': optimal_weights.tolist(),
            'model_names': [m['name'] for m in models_config]
        }, f, indent=2)

    print(f"✓ Pesos óptimos guardados en: {weights_file}")

    print("\n" + "="*80)
    print("✓ EXPERIMENTO 3 COMPLETADO")
    print("="*80)

if __name__ == "__main__":
    main()
