#!/usr/bin/env python3
"""
Script de comparación de todos los experimentos realizados.
Genera tabla comparativa y visualización de resultados.
"""

import json
import matplotlib
matplotlib.use('Agg')  # Backend sin interfaz gráfica
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def load_experiment_results():
    """Carga resultados de todos los experimentos"""
    base_dir = Path(__file__).parent.parent.parent
    models_dir = base_dir / "models"

    results = []

    # Experimento 1: Embeddings + LogReg
    exp01_dir = models_dir / "exp01_embeddings_logreg"
    if (exp01_dir / "summary.json").exists():
        with open(exp01_dir / "summary.json") as f:
            exp01 = json.load(f)

        for model_result in exp01['all_results']:
            results.append({
                'Experimento': 'Exp1: Embeddings+LogReg',
                'Modelo': model_result['model_name'].split('/')[-1],
                'F1 (dev)': model_result['dev_metrics']['f1'],
                'Accuracy': model_result['dev_metrics']['accuracy'],
                'Precision': model_result['dev_metrics']['precision'],
                'Recall': model_result['dev_metrics']['recall'],
                'Latencia (ms)': model_result['timings'].get('latency_ms_per_sentence', 0)
            })

    # Experimento 2: BERT Fine-tuning
    exp02_dir = models_dir / "exp02_bert_finetuning"
    if (exp02_dir / "summary.json").exists():
        with open(exp02_dir / "summary.json") as f:
            exp02 = json.load(f)

        for model_result in exp02['all_results']:
            results.append({
                'Experimento': 'Exp2: BERT Fine-tuning',
                'Modelo': model_result['model_name'].split('/')[-1],
                'F1 (dev)': model_result['dev_metrics']['f1'],
                'Accuracy': model_result['dev_metrics']['accuracy'],
                'Precision': model_result['dev_metrics']['precision'],
                'Recall': model_result['dev_metrics']['recall'],
                'Latencia (ms)': 0  # No medimos latencia en BERT
            })

    # Experimento 3: Ensemble
    exp03_dir = models_dir / "exp03_ensemble"
    if (exp03_dir / "results.json").exists():
        with open(exp03_dir / "results.json") as f:
            exp03 = json.load(f)

        results.append({
            'Experimento': 'Exp3: Ensemble',
            'Modelo': 'Soft Voting (3 modelos)',
            'F1 (dev)': exp03['ensemble_metrics']['f1'],
            'Accuracy': 0,  # No calculamos individualmente
            'Precision': 0,
            'Recall': 0,
            'Latencia (ms)': 0
        })

    return results

def create_comparison_table(results):
    """Crea tabla comparativa"""
    df = pd.DataFrame(results)

    # Ordenar por F1
    df = df.sort_values('F1 (dev)', ascending=False)

    return df

def create_visualizations(df, output_dir):
    """Crea visualizaciones comparativas"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figura 1: Comparación de F1 scores
    plt.figure(figsize=(12, 6))
    colors = ['#2ecc71' if f1 >= 0.85 else '#e74c3c' for f1 in df['F1 (dev)']]

    bars = plt.barh(range(len(df)), df['F1 (dev)'], color=colors)
    plt.yticks(range(len(df)), [f"{row['Modelo'][:30]}\n({row['Experimento']})"
                                  for _, row in df.iterrows()], fontsize=8)
    plt.xlabel('F1 Score (Dev Set)', fontsize=12)
    plt.title('Comparación de Modelos - F1 Score', fontsize=14, fontweight='bold')
    plt.axvline(x=0.85, color='orange', linestyle='--', linewidth=2, label='Target F1=0.85')
    plt.xlim(0, 1.05)
    plt.grid(axis='x', alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_dir / "model_comparison_f1.png", dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica guardada: {output_dir / 'model_comparison_f1.png'}")
    plt.close()

    # Figura 2: Precision vs Recall (solo para modelos con datos)
    df_with_metrics = df[(df['Precision'] > 0) & (df['Recall'] > 0)]

    if len(df_with_metrics) > 0:
        plt.figure(figsize=(10, 8))

        for exp in df_with_metrics['Experimento'].unique():
            df_exp = df_with_metrics[df_with_metrics['Experimento'] == exp]
            plt.scatter(df_exp['Recall'], df_exp['Precision'],
                       s=200, alpha=0.6, label=exp)

            for _, row in df_exp.iterrows():
                plt.annotate(row['Modelo'][:20], (row['Recall'], row['Precision']),
                           fontsize=8, ha='center')

        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision vs Recall por Experimento', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xlim(0.8, 1.05)
        plt.ylim(0.8, 1.05)
        plt.legend()
        plt.tight_layout()

        plt.savefig(output_dir / "precision_recall_scatter.png", dpi=300, bbox_inches='tight')
        print(f"✓ Gráfica guardada: {output_dir / 'precision_recall_scatter.png'}")
        plt.close()

def main():
    base_dir = Path(__file__).parent.parent.parent
    eval_dir = base_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("COMPARACIÓN DE TODOS LOS EXPERIMENTOS")
    print("="*80)

    # Cargar resultados
    print("\nCargando resultados de experimentos...")
    results = load_experiment_results()

    if not results:
        print("✗ No se encontraron resultados de experimentos")
        return

    print(f"✓ {len(results)} modelos encontrados")

    # Crear tabla comparativa
    df = create_comparison_table(results)

    # Mostrar tabla
    print("\n" + "="*80)
    print("TABLA COMPARATIVA DE MODELOS")
    print("="*80)
    print()

    # Formatear para mejor visualización
    pd.set_option('display.max_colwidth', 40)
    pd.set_option('display.width', 120)

    print(df.to_string(index=False))

    # Guardar tabla como CSV
    csv_path = eval_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Tabla guardada en: {csv_path}")

    # Crear visualizaciones
    print("\nGenerando visualizaciones...")
    create_visualizations(df, eval_dir)

    # Resumen estadístico
    print("\n" + "="*80)
    print("RESUMEN ESTADÍSTICO")
    print("="*80)

    print(f"\nNúmero total de modelos evaluados: {len(df)}")
    print(f"F1 promedio: {df['F1 (dev)'].mean():.4f}")
    print(f"F1 máximo: {df['F1 (dev)'].max():.4f}")
    print(f"F1 mínimo: {df['F1 (dev)'].min():.4f}")

    # Mejor modelo
    best_model = df.iloc[0]
    print(f"\nMEJOR MODELO:")
    print(f"  Experimento: {best_model['Experimento']}")
    print(f"  Modelo: {best_model['Modelo']}")
    print(f"  F1: {best_model['F1 (dev)']:.4f}")

    # Modelos que superan el target
    target_met = df[df['F1 (dev)'] >= 0.85]
    print(f"\nModelos que superan F1 >= 0.85: {len(target_met)}/{len(df)}")

    print("\n" + "="*80)
    print("✓ COMPARACIÓN COMPLETADA")
    print("="*80)

if __name__ == "__main__":
    main()
