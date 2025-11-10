"""
Integra variantes generadas por LLM al dataset principal
"""
import json
from pathlib import Path

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    base_path = Path("data/annotations")

    # Cargar dataset principal
    main_data = load_jsonl(base_path / "all_annotations.jsonl")
    print(f"Dataset principal: {len(main_data)} anotaciones")

    # Cargar variantes LLM
    llm_variants_path = base_path / "llm_generated_variants.jsonl"
    if not llm_variants_path.exists():
        print(f"⚠️  No se encontró {llm_variants_path}")
        return

    llm_data = load_jsonl(llm_variants_path)
    print(f"Variantes LLM: {len(llm_data)} anotaciones")

    # Verificar si ya están integradas
    main_texts = {item['text'] for item in main_data}
    new_llm = [item for item in llm_data if item['text'] not in main_texts]

    print(f"Nuevas variantes LLM a agregar: {len(new_llm)}")

    if len(new_llm) == 0:
        print("✓ Todas las variantes LLM ya están integradas")
        return

    # Backup
    backup_path = base_path / "all_annotations_before_llm.jsonl"
    if not backup_path.exists():
        save_jsonl(main_data, backup_path)
        print(f"✓ Backup guardado en: {backup_path}")

    # Integrar
    combined_data = main_data + new_llm
    print(f"Dataset combinado: {len(combined_data)} anotaciones")

    # Guardar
    save_jsonl(combined_data, base_path / "all_annotations.jsonl")
    print(f"✓ Dataset actualizado")

    # Estadísticas
    label_counts = {}
    for item in combined_data:
        label = item.get('label', 'UNKNOWN')
        label_counts[label] = label_counts.get(label, 0) + 1

    print("\nEstadísticas:")
    total = len(combined_data)
    for label, count in label_counts.items():
        pct = count / total * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

if __name__ == "__main__":
    main()
