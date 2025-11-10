#!/usr/bin/env python3
"""
Experimento 2 MEJORADO: BERT Fine-tuning con dropout y regularización
Mejoras:
- Hidden dropout
- Attention dropout
- Label smoothing
- Warmup + cosine decay
- Gradient clipping
"""

import json
import os
import time
import numpy as np
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Modelos a probar
BERT_MODELS = [
    "dccuchile/bert-base-spanish-wwm-cased",
    "bert-base-multilingual-cased"
]

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def prepare_dataset(data, tokenizer, max_length=128):
    texts = [d['text'] for d in data]
    labels = [1 if d['label'] == 'TAREA' else 0 for d in data]

    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )

    dataset_dict = {
        'input_ids': encodings['input_ids'].tolist(),
        'attention_mask': encodings['attention_mask'].tolist(),
        'labels': labels
    }

    return Dataset.from_dict(dataset_dict)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, predictions)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_and_evaluate_bert_improved(
    model_name: str,
    train_dataset,
    dev_dataset,
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    # NUEVOS PARÁMETROS
    hidden_dropout: float = 0.2,
    attention_dropout: float = 0.2,
    label_smoothing: float = 0.1,
    warmup_ratio: float = 0.1
):
    """
    Fine-tuning mejorado con dropout y regularización
    """
    print(f"\n{'='*80}")
    print(f"Fine-tuning MEJORADO: {model_name}")
    print(f"{'='*80}")
    print(f"\nMejoras aplicadas:")
    print(f"  - Hidden dropout: {hidden_dropout}")
    print(f"  - Attention dropout: {attention_dropout}")
    print(f"  - Label smoothing: {label_smoothing}")
    print(f"  - Warmup ratio: {warmup_ratio}")
    print(f"  - Gradient clipping: 1.0")
    print(f"  - Cosine LR decay")

    model_output_dir = output_dir / model_name.replace('/', '_')
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # CARGAR MODELO CON DROPOUT MEJORADO
    # ========================================================================

    print("\nCargando modelo con configuración mejorada...")
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Cargar config y modificar dropouts
    config = AutoConfig.from_pretrained(model_name)
    config.hidden_dropout_prob = hidden_dropout
    config.attention_probs_dropout_prob = attention_dropout
    config.num_labels = 2
    config.problem_type = "single_label_classification"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config
    )

    load_time = time.time() - start_time
    print(f"✓ Modelo cargado en {load_time:.2f}s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    # ========================================================================
    # TRAINING ARGUMENTS MEJORADOS
    # ========================================================================

    training_args = TrainingArguments(
        output_dir=str(model_output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,

        # MEJORAS DE REGULARIZACIÓN
        weight_decay=0.01,
        max_grad_norm=1.0,              # Gradient clipping
        label_smoothing_factor=label_smoothing,

        # MEJORAS DE LR SCHEDULING
        warmup_ratio=warmup_ratio,
        lr_scheduler_type='cosine',     # Cosine decay > linear

        # MEJORAS DE OPTIMIZACIÓN
        gradient_accumulation_steps=2,  # Simular batch_size=32

        # Configuración estándar
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_dir=str(model_output_dir / "logs"),
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=42
    )

    # ========================================================================
    # ENTRENAR
    # ========================================================================

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("\nIniciando fine-tuning...")
    start_time = time.time()
    train_result = trainer.train()
    train_time = time.time() - start_time

    print(f"\n✓ Fine-tuning completado en {train_time:.2f}s")
    print(f"Loss final: {train_result.training_loss:.4f}")

    # ========================================================================
    # EVALUAR
    # ========================================================================

    print("\nEvaluando en dev set...")
    eval_result = trainer.evaluate()

    print(f"\nMétricas en DEV:")
    print(f"  Accuracy:  {eval_result['eval_accuracy']:.4f}")
    print(f"  Precision: {eval_result['eval_precision']:.4f}")
    print(f"  Recall:    {eval_result['eval_recall']:.4f}")
    print(f"  F1:        {eval_result['eval_f1']:.4f}")

    # Matriz de confusión
    predictions = trainer.predict(dev_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nMatriz de confusión:")
    print(cm)

    # ========================================================================
    # GUARDAR MODELO
    # ========================================================================

    print(f"\nGuardando modelo...")
    trainer.save_model(str(model_output_dir / "best_model"))
    tokenizer.save_pretrained(str(model_output_dir / "best_model"))

    # Guardar resultados
    results = {
        'model_name': model_name,
        'improvements': {
            'hidden_dropout': hidden_dropout,
            'attention_dropout': attention_dropout,
            'label_smoothing': label_smoothing,
            'warmup_ratio': warmup_ratio,
            'gradient_clipping': 1.0,
            'lr_scheduler': 'cosine',
            'gradient_accumulation_steps': 2
        },
        'training_args': {
            'epochs': epochs,
            'batch_size': batch_size,
            'effective_batch_size': batch_size * 2,  # Con accumulation
            'learning_rate': learning_rate
        },
        'train_loss': float(train_result.training_loss),
        'dev_metrics': {
            'accuracy': float(eval_result['eval_accuracy']),
            'precision': float(eval_result['eval_precision']),
            'recall': float(eval_result['eval_recall']),
            'f1': float(eval_result['eval_f1'])
        },
        'confusion_matrix': cm.tolist(),
        'timings': {
            'load_time': load_time,
            'train_time': train_time
        }
    }

    with open(model_output_dir / "results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("✓ Modelo y resultados guardados")

    # Limpiar memoria
    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results

def main():
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "data" / "splits"
    output_dir = base_dir / "models" / "exp02_bert_improved"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("EXPERIMENTO 2 MEJORADO: BERT FINE-TUNING")
    print("="*80)

    # Cargar datos
    print("\nCargando datasets...")
    train_data = load_jsonl(data_dir / "train.jsonl")
    dev_data = load_jsonl(data_dir / "dev.jsonl")

    print(f"Train: {len(train_data)} oraciones")
    print(f"Dev: {len(dev_data)} oraciones")

    # Entrenar cada modelo
    all_results = []

    for model_name in BERT_MODELS:
        try:
            print(f"\nPreparando datasets para {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            train_dataset = prepare_dataset(train_data, tokenizer)
            dev_dataset = prepare_dataset(dev_data, tokenizer)

            # Fine-tuning mejorado
            results = train_and_evaluate_bert_improved(
                model_name,
                train_dataset,
                dev_dataset,
                output_dir,
                epochs=5,
                batch_size=16,
                learning_rate=2e-5,
                hidden_dropout=0.2,
                attention_dropout=0.2,
                label_smoothing=0.1,
                warmup_ratio=0.1
            )

            all_results.append(results)

        except Exception as e:
            print(f"\n✗ Error con {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ========================================================================
    # RESUMEN
    # ========================================================================

    print("\n" + "="*80)
    print("RESUMEN DE RESULTADOS")
    print("="*80)

    best_model = None
    best_f1 = 0

    for i, result in enumerate(all_results, 1):
        f1 = result['dev_metrics']['f1']
        print(f"\n{i}. {result['model_name']}")
        print(f"   F1: {f1:.4f}")
        print(f"   Accuracy: {result['dev_metrics']['accuracy']:.4f}")
        print(f"   Train time: {result['timings']['train_time']:.2f}s")

        if f1 > best_f1:
            best_f1 = f1
            best_model = result

    # Guardar resumen
    summary = {
        'experiment': 'exp02_bert_improved',
        'models_tested': len(all_results),
        'all_results': all_results,
        'best_model': best_model['model_name'] if best_model else None,
        'best_f1': float(best_f1)
    }

    with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Mejor modelo
    if best_model:
        print("\n" + "="*80)
        print("MEJOR MODELO")
        print("="*80)
        print(f"Modelo: {best_model['model_name']}")
        print(f"F1: {best_f1:.4f}")
        print(f"Target alcanzado: {'✓ SÍ' if best_f1 >= 0.85 else '✗ NO'}")

    print("\n" + "="*80)
    print("✓ EXPERIMENTO 2 MEJORADO COMPLETADO")
    print("="*80)

if __name__ == "__main__":
    main()
