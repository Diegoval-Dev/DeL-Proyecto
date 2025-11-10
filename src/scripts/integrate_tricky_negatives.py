"""
Script para integrar negativos difíciles al dataset principal
"""
import json
from pathlib import Path

def load_jsonl(path):
    """Carga archivo JSONL"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    """Guarda datos en JSONL"""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    base_path = Path("data/annotations")

    # Cargar dataset principal
    main_data = load_jsonl(base_path / "all_annotations.jsonl")
    print(f"Dataset principal: {len(main_data)} anotaciones")

    # Cargar negativos difíciles
    tricky_data = load_jsonl(base_path / "tricky_negatives.jsonl")
    print(f"Negativos difíciles: {len(tricky_data)} anotaciones")

    # Verificar si ya están integrados
    tricky_texts = {item['text'] for item in tricky_data}
    main_texts = {item['text'] for item in main_data}

    # Filtrar solo los nuevos
    new_tricky = [item for item in tricky_data if item['text'] not in main_texts]
    print(f"Nuevos negativos a agregar: {len(new_tricky)}")

    if len(new_tricky) == 0:
        print("✓ Todos los negativos difíciles ya están integrados")
        return

    # Backup del dataset original
    backup_path = base_path / "all_annotations_before_tricky.jsonl"
    if not backup_path.exists():
        save_jsonl(main_data, backup_path)
        print(f"✓ Backup guardado en: {backup_path}")

    # Integrar nuevos negativos
    combined_data = main_data + new_tricky
    print(f"Dataset combinado: {len(combined_data)} anotaciones")

    # Guardar dataset actualizado
    save_jsonl(combined_data, base_path / "all_annotations.jsonl")
    print(f"✓ Dataset actualizado guardado en: {base_path / 'all_annotations.jsonl'}")

    # Estadísticas
    label_counts = {}
    for item in combined_data:
        label = item.get('label', 'UNKNOWN')
        label_counts[label] = label_counts.get(label, 0) + 1

    print("\nEstadísticas del dataset:")
    total = len(combined_data)
    for label, count in label_counts.items():
        pct = count / total * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

if __name__ == "__main__":
    main()
