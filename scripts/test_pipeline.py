#!/usr/bin/env python3
"""
Script de prueba rápida del pipeline completo
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocess import clean_text
from sentence_split import split_sentences
from infer_classifier import SentenceTaskClassifier
from ner_extract import extract_person_responsable
from date_extract import extract_date_iso

def test_pipeline():
    print("="*80)
    print("TEST DEL PIPELINE COMPLETO")
    print("="*80)

    # Texto de prueba
    text = """
    Juan debe enviar el informe antes del viernes 15 de noviembre.
    Se discutió el presupuesto del proyecto.
    María coordinará la reunión con el equipo técnico el próximo martes.
    Los resultados fueron presentados por el director.
    Carlos tiene que revisar la documentación antes del 20 de noviembre.
    """

    print("\n1. PREPROCESAMIENTO")
    print("-" * 80)
    cleaned = clean_text(text)
    print(f"Texto limpio ({len(cleaned)} caracteres)")

    print("\n2. SEGMENTACIÓN EN ORACIONES")
    print("-" * 80)
    sentences = split_sentences(cleaned)
    print(f"Oraciones detectadas: {len(sentences)}")
    for i, s in enumerate(sentences, 1):
        print(f"  {i}. {s[:60]}...")

    print("\n3. CARGA DEL MODELO CLASIFICADOR")
    print("-" * 80)
    base_dir = Path(__file__).parent.parent
    clf = SentenceTaskClassifier(model_dir=base_dir / "models" / "best_baseline")

    print("\n4. CLASIFICACIÓN Y EXTRACCIÓN")
    print("-" * 80)
    base_date = "2025-11-08"

    for i, sent in enumerate(sentences, 1):
        is_task, score = clf.predict_sentence(sent)

        if is_task:
            responsable = extract_person_responsable(sent)
            fecha = extract_date_iso(sent, base_date)

            print(f"\n✅ TAREA #{i} (score: {score:.3f})")
            print(f"   Oración: {sent}")
            print(f"   Responsable: {responsable}")
            print(f"   Fecha: {fecha}")
        else:
            print(f"\n❌ NO TAREA #{i} (score: {score:.3f})")
            print(f"   Oración: {sent[:80]}...")

    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*80)

if __name__ == "__main__":
    try:
        test_pipeline()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
