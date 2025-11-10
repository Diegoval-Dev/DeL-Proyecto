#!/usr/bin/env python3
"""
Experimento 3: Ensemble de mejores modelos
Combina predicciones de los mejores modelos usando soft voting.
Target: F1 >= 0.85
"""

import json
import joblib
import numpy as np
from pathlib import Path
import torch
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

    # Generar embeddings
    X = encoder.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # Probabilidades
    probas = classifier.predict_proba(X)
    return probas

def predict_bert_model(model_dir, texts):
    """Predicciones con modelo BERT fine-tuned"""
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    probas_list = []

    # Procesar en batches
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        # Tokenizar
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        # Predicciones
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probas = torch.softmax(logits, dim=-1).numpy()

        probas_list.append(probas)

    # Concatenar todas las probabilidades
    all_probas = np.vstack(probas_list)
    return all_probas

def ensemble_soft_voting(models_probas):
    """Ensemble por promedio de probabilidades (soft voting)"""
    # Promedio de probabilidades
    avg_probas = np.mean(models_probas, axis=0)
    # Predicciones finales
    predictions = np.argmax(avg_probas, axis=1)
    return predictions, avg_probas

def main():
    # Rutas
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "data" / "splits"
    output_dir = base_dir / "models" / "exp03_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("EXPERIMENTO 3: ENSEMBLE DE MODELOS")
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
            'name': 'BERT (dccuchile)',
            'type': 'bert',
            'path': base_dir / "models" / "exp02_bert_finetuning" / "dccuchile_bert-base-spanish-wwm-cased" / "best_model"
        },
        {
            'name': 'BERT (multilingual)',
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

            # Predicciones individuales
            preds = np.argmax(probas, axis=1)
            f1 = f1_score(y_true, preds)

            print(f"F1 individual: {f1:.4f}")

            all_probas.append(probas)
            individual_results.append({
                'name': model_info['name'],
                'f1': float(f1),
                'predictions': preds.tolist()
            })

        except Exception as e:
            print(f"✗ Error cargando {model_info['name']}: {e}")
            continue

    # Ensemble
    if len(all_probas) >= 2:
        print(f"\n{'='*80}")
        print("ENSEMBLE (Soft Voting)")
        print(f"{'='*80}")

        ensemble_preds, ensemble_probas = ensemble_soft_voting(all_probas)

        # Métricas del ensemble
        f1 = f1_score(y_true, ensemble_preds)

        print(f"\nF1 Ensemble: {f1:.4f}")
        print(f"\nReporte de clasificación:")
        print(classification_report(
            y_true,
            ensemble_preds,
            target_names=['NO_TAREA', 'TAREA'],
            digits=4
        ))

        cm = confusion_matrix(y_true, ensemble_preds)
        print(f"\nMatriz de confusión:")
        print(cm)

        # Guardar resultados
        results = {
            'experiment': 'exp03_ensemble',
            'ensemble_method': 'soft_voting',
            'num_models': len(all_probas),
            'individual_models': individual_results,
            'ensemble_metrics': {
                'f1': float(f1),
                'confusion_matrix': cm.tolist()
            }
        }

        with open(output_dir / "results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Comparación
        print(f"\n{'='*80}")
        print("COMPARACIÓN")
        print(f"{'='*80}")

        for res in individual_results:
            print(f"{res['name']}: F1 = {res['f1']:.4f}")

        print(f"Ensemble: F1 = {f1:.4f}")

        best_individual = max(individual_results, key=lambda x: x['f1'])
        improvement = f1 - best_individual['f1']

        print(f"\nMejora sobre mejor individual: {improvement:+.4f}")
        print(f"Target alcanzado: {'✓ SÍ' if f1 >= 0.85 else '✗ NO'} (target: 0.85)")

    else:
        print("\n✗ No hay suficientes modelos para ensemble (mínimo 2)")

    print("\n" + "="*80)
    print("✓ EXPERIMENTO 3 COMPLETADO")
    print("="*80)

if __name__ == "__main__":
    main()
