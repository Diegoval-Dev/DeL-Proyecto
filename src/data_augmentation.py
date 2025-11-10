"""
Data Augmentation para aumentar dataset de entrenamiento
Técnicas: Synonym replacement y Back-translation
"""

import random
import re
from typing import List, Tuple
import json

# Para back-translation necesitaremos MarianMT (ligero)
try:
    from transformers import MarianMTModel, MarianTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ transformers no disponible, solo synonym augmentation")

# Sinónimos comunes en español para verbos de acción
SYNONYM_DICT = {
    'enviar': ['remitir', 'mandar', 'despachar', 'transmitir'],
    'preparar': ['elaborar', 'confeccionar', 'desarrollar', 'crear'],
    'revisar': ['verificar', 'chequear', 'examinar', 'inspeccionar'],
    'coordinar': ['organizar', 'gestionar', 'planificar', 'dirigir'],
    'entregar': ['proveer', 'suministrar', 'facilitar', 'proporcionar'],
    'completar': ['finalizar', 'terminar', 'concluir', 'acabar'],
    'actualizar': ['renovar', 'modernizar', 'refrescar', 'poner al día'],
    'documentar': ['registrar', 'anotar', 'describir', 'detallar'],
    'notificar': ['avisar', 'informar', 'comunicar', 'advertir'],
    'resolver': ['solucionar', 'arreglar', 'subsanar', 'remediar'],
    'validar': ['verificar', 'confirmar', 'ratificar', 'corroborar'],
    'debe': ['tiene que', 'necesita', 'requiere', 'está obligado a'],
    'puede': ['es capaz de', 'está en condiciones de', 'tiene la posibilidad de'],
    'reunión': ['junta', 'encuentro', 'sesión', 'conferencia'],
    'informe': ['reporte', 'documento', 'escrito', 'memoria'],
    'antes': ['previo a', 'con anterioridad a', 'anticipadamente'],
    'después': ['posteriormente', 'luego', 'más tarde', 'a continuación'],
}

class SpanishAugmenter:
    """Aumentador de datos para español"""

    def __init__(self, use_backtranslation=True):
        self.use_backtranslation = use_backtranslation and TRANSFORMERS_AVAILABLE

        if self.use_backtranslation:
            print("Cargando modelos de back-translation...")
            # Español → Inglés
            self.es_en_tokenizer = MarianTokenizer.from_pretrained(
                'Helsinki-NLP/opus-mt-es-en'
            )
            self.es_en_model = MarianMTModel.from_pretrained(
                'Helsinki-NLP/opus-mt-es-en'
            )

            # Inglés → Español
            self.en_es_tokenizer = MarianTokenizer.from_pretrained(
                'Helsinki-NLP/opus-mt-en-es'
            )
            self.en_es_model = MarianMTModel.from_pretrained(
                'Helsinki-NLP/opus-mt-en-es'
            )
            print("✓ Modelos cargados")

    def synonym_replacement(self, text: str, n_replacements: int = 2) -> str:
        """
        Reemplaza n_replacements palabras con sinónimos
        """
        words = text.split()
        words_lower = [w.lower() for w in words]

        # Encontrar palabras reemplazables
        replaceable_indices = [
            i for i, w in enumerate(words_lower)
            if w in SYNONYM_DICT
        ]

        if not replaceable_indices:
            return text

        # Seleccionar aleatoriamente cuáles reemplazar
        n_to_replace = min(n_replacements, len(replaceable_indices))
        indices_to_replace = random.sample(replaceable_indices, n_to_replace)

        # Reemplazar
        augmented_words = words.copy()
        for idx in indices_to_replace:
            original_word = words_lower[idx]
            synonyms = SYNONYM_DICT[original_word]
            new_word = random.choice(synonyms)

            # Preservar capitalización
            if words[idx][0].isupper():
                new_word = new_word.capitalize()

            augmented_words[idx] = new_word

        return ' '.join(augmented_words)

    def back_translate(self, text: str) -> str:
        """
        Traduce español → inglés → español para parafrasear
        """
        if not self.use_backtranslation:
            return text

        try:
            # Español → Inglés
            inputs = self.es_en_tokenizer(text, return_tensors="pt", padding=True)
            translated = self.es_en_model.generate(**inputs, max_length=128)
            english_text = self.es_en_tokenizer.decode(
                translated[0], skip_special_tokens=True
            )

            # Inglés → Español
            inputs = self.en_es_tokenizer(english_text, return_tensors="pt", padding=True)
            back_translated = self.en_es_model.generate(**inputs, max_length=128)
            spanish_text = self.en_es_tokenizer.decode(
                back_translated[0], skip_special_tokens=True
            )

            return spanish_text

        except Exception as e:
            print(f"⚠️ Error en back-translation: {e}")
            return text

    def augment_sentence(self, text: str, method: str = 'synonym') -> str:
        """
        Aumenta una oración con el método especificado

        Args:
            text: Texto a aumentar
            method: 'synonym', 'backtranslate', o 'both'
        """
        if method == 'synonym':
            return self.synonym_replacement(text, n_replacements=2)

        elif method == 'backtranslate':
            return self.back_translate(text)

        elif method == 'both':
            # Aplicar ambos
            text = self.synonym_replacement(text, n_replacements=1)
            text = self.back_translate(text)
            return text

        return text

def augment_dataset(
    texts: List[str],
    labels: List[int],
    augment_factor: int = 2,
    use_backtranslation: bool = True,
    balance_classes: bool = True
) -> Tuple[List[str], List[int]]:
    """
    Aumenta dataset completo

    Args:
        texts: Lista de textos
        labels: Lista de labels (0 o 1)
        augment_factor: Cuántas versiones generar por texto
        use_backtranslation: Si usar back-translation (más lento)
        balance_classes: Si balancear clases TAREA/NO_TAREA

    Returns:
        Tupla (textos_aumentados, labels_aumentados)
    """
    print(f"\n{'='*80}")
    print("DATA AUGMENTATION")
    print(f"{'='*80}")
    print(f"Dataset original: {len(texts)} oraciones")
    print(f"Factor de aumento: {augment_factor}x")
    print(f"Back-translation: {'✓ Sí' if use_backtranslation else '✗ No'}")

    augmenter = SpanishAugmenter(use_backtranslation=use_backtranslation)

    augmented_texts = []
    augmented_labels = []

    # Si balance_classes, aumentar más la clase minoritaria
    if balance_classes:
        n_tarea = sum(labels)
        n_no_tarea = len(labels) - n_tarea

        tarea_factor = augment_factor
        no_tarea_factor = augment_factor

        # Aumentar más la clase minoritaria
        if n_tarea < n_no_tarea:
            tarea_factor = int(augment_factor * 1.5)
        elif n_no_tarea < n_tarea:
            no_tarea_factor = int(augment_factor * 1.5)

        print(f"\nBalance de clases:")
        print(f"  TAREA: {n_tarea} → factor {tarea_factor}x")
        print(f"  NO_TAREA: {n_no_tarea} → factor {no_tarea_factor}x")

    # Procesar cada texto
    for i, (text, label) in enumerate(zip(texts, labels)):
        # Agregar original
        augmented_texts.append(text)
        augmented_labels.append(label)

        # Determinar factor para este ejemplo
        if balance_classes:
            factor = tarea_factor if label == 1 else no_tarea_factor
        else:
            factor = augment_factor

        # Generar variantes
        for j in range(factor):
            # Alternar métodos
            if j % 2 == 0 or not use_backtranslation:
                method = 'synonym'
            else:
                method = 'backtranslate'

            try:
                augmented_text = augmenter.augment_sentence(text, method=method)

                # Solo agregar si es diferente del original
                if augmented_text != text and len(augmented_text.strip()) > 10:
                    augmented_texts.append(augmented_text)
                    augmented_labels.append(label)

            except Exception as e:
                print(f"⚠️ Error aumentando texto {i}: {e}")
                continue

        # Progress
        if (i + 1) % 50 == 0:
            print(f"Procesados: {i+1}/{len(texts)}")

    print(f"\n✓ Dataset aumentado: {len(augmented_texts)} oraciones")
    print(f"  Aumento real: {len(augmented_texts) / len(texts):.1f}x")

    # Estadísticas finales
    final_tarea = sum(augmented_labels)
    final_no_tarea = len(augmented_labels) - final_tarea
    print(f"\nBalance final:")
    print(f"  TAREA: {final_tarea} ({100*final_tarea/len(augmented_labels):.1f}%)")
    print(f"  NO_TAREA: {final_no_tarea} ({100*final_no_tarea/len(augmented_labels):.1f}%)")

    return augmented_texts, augmented_labels

def save_augmented_data(texts: List[str], labels: List[int], output_path: str):
    """Guarda datos aumentados en formato JSONL"""
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for text, label in zip(texts, labels):
            entry = {
                'text': text,
                'label': 'TAREA' if label == 1 else 'NO_TAREA'
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"✓ Datos guardados en: {output_path}")

# Testing
if __name__ == "__main__":
    # Ejemplo de uso
    test_texts = [
        "Juan debe enviar el informe antes del viernes",
        "Se discutió el presupuesto del proyecto",
        "María coordinará la reunión técnica"
    ]
    test_labels = [1, 0, 1]

    print("="*80)
    print("TEST DE DATA AUGMENTATION")
    print("="*80)

    augmented_texts, augmented_labels = augment_dataset(
        test_texts,
        test_labels,
        augment_factor=2,
        use_backtranslation=False  # Solo sinónimos para test rápido
    )

    print("\nEjemplos generados:")
    for i, (text, label) in enumerate(zip(augmented_texts[:10], augmented_labels[:10])):
        print(f"{i+1}. [{label}] {text}")
