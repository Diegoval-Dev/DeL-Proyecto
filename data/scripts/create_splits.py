#!/usr/bin/env python3
"""
Script para dividir el dataset anotado en train/dev/test splits.
Divide por documento (no por oración) para evitar data leakage.
Split ratio: 70% train / 15% dev / 15% test
"""

import json
import random
from pathlib import Path
from collections import defaultdict

def load_annotations(path):
    """Carga anotaciones desde archivo JSONL"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def group_by_document(annotations):
    """Agrupa oraciones por documento"""
    docs = defaultdict(list)
    for ann in annotations:
        docs[ann['doc_id']].append(ann)
    return docs

def stratified_split_by_doc(docs, train_ratio=0.70, dev_ratio=0.15):
    """
    Divide documentos en train/dev/test manteniendo balance de clases.

    Args:
        docs: Dict de doc_id -> lista de anotaciones
        train_ratio: Proporción para entrenamiento
        dev_ratio: Proporción para desarrollo

    Returns:
        train_docs, dev_docs, test_docs (listas de doc_ids)
    """
    # Calcular balance TAREA/NO_TAREA por documento
    doc_stats = {}
    for doc_id, sentences in docs.items():
        tarea_count = sum(1 for s in sentences if s['label'] == 'TAREA')
        total = len(sentences)
        doc_stats[doc_id] = tarea_count / total if total > 0 else 0

    # Ordenar documentos por su ratio de TAREA
    sorted_docs = sorted(doc_stats.items(), key=lambda x: x[1])
    doc_ids = [doc_id for doc_id, _ in sorted_docs]

    # Shuffle con seed fija para reproducibilidad
    random.seed(42)
    random.shuffle(doc_ids)

    # Calcular tamaños de splits
    total_docs = len(doc_ids)
    train_size = int(total_docs * train_ratio)
    dev_size = int(total_docs * dev_ratio)

    # Dividir
    train_docs = doc_ids[:train_size]
    dev_docs = doc_ids[train_size:train_size + dev_size]
    test_docs = doc_ids[train_size + dev_size:]

    return train_docs, dev_docs, test_docs

def save_split(docs, doc_ids, output_path):
    """Guarda un split a archivo JSONL"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sentences = []
    for doc_id in doc_ids:
        sentences.extend(docs[doc_id])

    with open(output_path, 'w', encoding='utf-8') as f:
        for sent in sentences:
            f.write(json.dumps(sent, ensure_ascii=False) + '\n')

    return sentences

def print_statistics(name, sentences):
    """Imprime estadísticas de un split"""
    total = len(sentences)
    tarea = sum(1 for s in sentences if s['label'] == 'TAREA')
    no_tarea = total - tarea

    with_responsable = sum(1 for s in sentences if 'responsable_gold' in s)
    with_fecha = sum(1 for s in sentences if 'fecha_gold' in s)

    print(f"\n{name}:")
    print(f"  Total oraciones: {total}")
    print(f"  TAREA: {tarea} ({100*tarea/total:.1f}%)")
    print(f"  NO_TAREA: {no_tarea} ({100*no_tarea/total:.1f}%)")
    print(f"  Con responsable_gold: {with_responsable}")
    print(f"  Con fecha_gold: {with_fecha}")

def main():
    # Rutas
    base_dir = Path(__file__).parent.parent
    annotations_path = base_dir / "annotations" / "all_annotations.jsonl"
    splits_dir = base_dir / "splits"

    print(f"Cargando anotaciones desde: {annotations_path}")
    annotations = load_annotations(annotations_path)
    print(f"Total de oraciones cargadas: {len(annotations)}")

    # Agrupar por documento
    docs = group_by_document(annotations)
    print(f"Total de documentos: {len(docs)}")

    # Crear splits
    print("\nCreando splits (70% train / 15% dev / 15% test)...")
    train_docs, dev_docs, test_docs = stratified_split_by_doc(docs)

    print(f"\nDocumentos por split:")
    print(f"  Train: {len(train_docs)} documentos")
    print(f"  Dev: {len(dev_docs)} documentos")
    print(f"  Test: {len(test_docs)} documentos")

    # Guardar splits
    print("\nGuardando splits...")
    train_sentences = save_split(docs, train_docs, splits_dir / "train.jsonl")
    dev_sentences = save_split(docs, dev_docs, splits_dir / "dev.jsonl")
    test_sentences = save_split(docs, test_docs, splits_dir / "test.jsonl")

    # Imprimir estadísticas
    print("\n" + "="*60)
    print("ESTADÍSTICAS DE SPLITS")
    print("="*60)

    print_statistics("TRAIN", train_sentences)
    print_statistics("DEV", dev_sentences)
    print_statistics("TEST", test_sentences)

    print("\n" + "="*60)
    print(f"✓ Splits creados exitosamente en: {splits_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
