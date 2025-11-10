"""
Calibración de probabilidades usando Platt Scaling
Mejora la confiabilidad de los scores de clasificación
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt
from typing import Tuple

def calibrate_classifier(
    base_classifier,
    X_val,
    y_val,
    method: str = 'sigmoid',
    return_best: bool = True
):
    """
    Calibra probabilidades de un clasificador entrenado

    Args:
        base_classifier: Clasificador ya entrenado
        X_val: Datos de validación (embeddings)
        y_val: Labels de validación
        method: 'sigmoid' (Platt) o 'isotonic'
        return_best: Si True, devuelve el mejor entre calibrado y no calibrado

    Returns:
        Clasificador calibrado (o el original si return_best=True y no mejora)
    """
    from sklearn.base import clone
    from sklearn.metrics import brier_score_loss

    print(f"\n{'='*80}")
    print("CALIBRACIÓN DE PROBABILIDADES")
    print(f"{'='*80}")
    print(f"Método: {method}")
    print(f"Datos de calibración: {len(X_val)} muestras")

    # Dividir datos de calibración en calibration y validation
    split_point = int(len(X_val) * 0.7)
    X_calib, X_val_test = X_val[:split_point], X_val[split_point:]
    y_calib, y_val_test = y_val[:split_point], y_val[split_point:]

    print(f"Usando {len(X_calib)} para calibración, {len(X_val_test)} para validación")

    # Intentar usar la nueva API primero
    try:
        from sklearn.calibration import FrozenEstimator

        # Crear clasificador calibrado con nueva API
        frozen = FrozenEstimator(base_classifier)
        calibrated = CalibratedClassifierCV(
            frozen,
            method=method,
            ensemble=True
        )
    except (ImportError, AttributeError):
        # Fallback a la API antigua (con warning)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            calibrated = CalibratedClassifierCV(
                base_classifier,
                method=method,
                cv='prefit',
                ensemble=True
            )

    # Calibrar
    print("\nCalibrando...")
    calibrated.fit(X_calib, y_calib)
    print("✓ Calibración completada")

    # Si return_best, comparar con el original
    if return_best and len(X_val_test) > 0:
        print("\nComparando modelos en datos de validación...")

        # Brier scores (menor es mejor)
        y_proba_orig = base_classifier.predict_proba(X_val_test)[:, 1]
        y_proba_cal = calibrated.predict_proba(X_val_test)[:, 1]

        brier_orig = brier_score_loss(y_val_test, y_proba_orig)
        brier_cal = brier_score_loss(y_val_test, y_proba_cal)

        print(f"  Brier Score original:  {brier_orig:.4f}")
        print(f"  Brier Score calibrado: {brier_cal:.4f}")

        if brier_cal < brier_orig:
            print("  ✓ Calibración mejora el modelo, usando versión calibrada")
            return calibrated
        else:
            print("  ⚠️ Calibración no mejora el modelo, usando versión original")
            return base_classifier

    return calibrated

def evaluate_calibration(
    classifier,
    X_test,
    y_test,
    n_bins: int = 10,
    output_path: str = None
) -> dict:
    """
    Evalúa la calibración del modelo

    Args:
        classifier: Clasificador a evaluar
        X_test: Datos de test
        y_test: Labels de test
        n_bins: Número de bins para curva de calibración
        output_path: Dónde guardar gráfica

    Returns:
        Dict con métricas de calibración
    """
    print(f"\n{'='*80}")
    print("EVALUACIÓN DE CALIBRACIÓN")
    print(f"{'='*80}")

    # Obtener probabilidades
    y_proba = classifier.predict_proba(X_test)[:, 1]

    # Calcular curva de calibración
    prob_true, prob_pred = calibration_curve(
        y_test,
        y_proba,
        n_bins=n_bins,
        strategy='uniform'
    )

    # Expected Calibration Error (ECE)
    ece = np.mean(np.abs(prob_true - prob_pred))

    # Maximum Calibration Error (MCE)
    mce = np.max(np.abs(prob_true - prob_pred))

    print(f"\nMétricas de calibración:")
    print(f"  Expected Calibration Error (ECE): {ece:.4f}")
    print(f"  Maximum Calibration Error (MCE): {mce:.4f}")
    print(f"  {'✓ Bien calibrado' if ece < 0.1 else '⚠️ Mal calibrado'}")

    # Visualización
    if output_path:
        plot_calibration_curve(
            prob_true,
            prob_pred,
            y_proba,
            y_test,
            ece,
            output_path
        )

    return {
        'ece': float(ece),
        'mce': float(mce),
        'prob_true': prob_true.tolist(),
        'prob_pred': prob_pred.tolist()
    }

def plot_calibration_curve(
    prob_true,
    prob_pred,
    y_proba,
    y_test,
    ece,
    output_path
):
    """Genera gráfica de curva de calibración"""
    from pathlib import Path

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Curva de calibración
    axes[0].plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    axes[0].plot(prob_pred, prob_true, 'o-', label='Model calibration')
    axes[0].set_xlabel('Mean predicted probability', fontsize=12)
    axes[0].set_ylabel('Fraction of positives', fontsize=12)
    axes[0].set_title(f'Calibration Curve (ECE={ece:.3f})', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Histograma de probabilidades
    axes[1].hist(y_proba[y_test == 0], bins=20, alpha=0.5, label='NO_TAREA', color='red')
    axes[1].hist(y_proba[y_test == 1], bins=20, alpha=0.5, label='TAREA', color='green')
    axes[1].set_xlabel('Predicted probability', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Distribution of Predicted Probabilities', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica guardada en: {output_path}")
    plt.close()

def compare_before_after_calibration(
    uncalibrated_clf,
    calibrated_clf,
    X_test,
    y_test,
    output_dir: str
):
    """
    Compara modelo antes y después de calibración
    """
    from pathlib import Path
    from sklearn.metrics import f1_score, brier_score_loss

    print(f"\n{'='*80}")
    print("COMPARACIÓN ANTES/DESPUÉS DE CALIBRACIÓN")
    print(f"{'='*80}")

    # Probabilidades antes
    y_proba_uncal = uncalibrated_clf.predict_proba(X_test)[:, 1]
    y_pred_uncal = uncalibrated_clf.predict(X_test)

    # Probabilidades después
    y_proba_cal = calibrated_clf.predict_proba(X_test)[:, 1]
    y_pred_cal = calibrated_clf.predict(X_test)

    # Métricas
    f1_uncal = f1_score(y_test, y_pred_uncal)
    f1_cal = f1_score(y_test, y_pred_cal)

    # Brier score (mide calibración)
    brier_uncal = brier_score_loss(y_test, y_proba_uncal)
    brier_cal = brier_score_loss(y_test, y_proba_cal)

    print(f"\nF1 Score:")
    print(f"  Sin calibrar: {f1_uncal:.4f}")
    print(f"  Calibrado:    {f1_cal:.4f}")
    print(f"  Cambio:       {f1_cal - f1_uncal:+.4f}")

    print(f"\nBrier Score (↓ mejor):")
    print(f"  Sin calibrar: {brier_uncal:.4f}")
    print(f"  Calibrado:    {brier_cal:.4f}")
    print(f"  Mejora:       {brier_uncal - brier_cal:+.4f}")

    # Evaluar calibración de ambos
    output_dir = Path(output_dir)

    eval_uncal = evaluate_calibration(
        uncalibrated_clf, X_test, y_test,
        output_path=output_dir / "calibration_before.png"
    )

    eval_cal = evaluate_calibration(
        calibrated_clf, X_test, y_test,
        output_path=output_dir / "calibration_after.png"
    )

    print(f"\nECE Comparison:")
    print(f"  Sin calibrar: {eval_uncal['ece']:.4f}")
    print(f"  Calibrado:    {eval_cal['ece']:.4f}")
    print(f"  Mejora:       {eval_uncal['ece'] - eval_cal['ece']:+.4f}")

    return {
        'uncalibrated': {
            'f1': float(f1_uncal),
            'brier_score': float(brier_uncal),
            'ece': eval_uncal['ece']
        },
        'calibrated': {
            'f1': float(f1_cal),
            'brier_score': float(brier_cal),
            'ece': eval_cal['ece']
        }
    }

# Testing
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    print("="*80)
    print("TEST DE CALIBRACIÓN")
    print("="*80)

    # Usar SVM que típicamente está mal calibrado
    # (a diferencia de LogisticRegression que ya está bien calibrado)
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42,
        flip_y=0.1  # Agregar algo de ruido
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    print(f"\nDatos: {len(X_train)} train, {len(X_cal)} calibration, {len(X_test)} test")

    # Entrenar SVM (típicamente mal calibrado)
    print("\nEntrenando SVM (típicamente requiere calibración)...")
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_train, y_train)

    # Calibrar con return_best=True
    print("\n" + "="*80)
    print("Intentando calibración con return_best=True")
    print("="*80)
    best_clf = calibrate_classifier(clf, X_cal, y_cal, method='sigmoid', return_best=True)

    # Determinar si devolvió el calibrado o el original
    is_calibrated = best_clf != clf

    print("\n" + "="*80)
    print(f"Modelo seleccionado: {'CALIBRADO' if is_calibrated else 'ORIGINAL'}")
    print("="*80)

    # Si se seleccionó calibrado, comparar
    if is_calibrated:
        results = compare_before_after_calibration(
            clf, best_clf, X_test, y_test,
            output_dir='eval/calibration_test'
        )
    else:
        print("\nModelo original ya estaba bien calibrado, no se aplicó calibración.")
        # Evaluar solo el original
        eval_result = evaluate_calibration(
            clf, X_test, y_test,
            output_path='eval/calibration_test/calibration_original.png'
        )

    print("\n✓ Test completado")
