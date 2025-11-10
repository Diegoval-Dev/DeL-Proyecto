# 🏗️ Arquitectura del Proyecto - ActionMiner Lite

**Versión**: 1.0.0
**Fecha de Baseline**: 2025-11-09
**Estado**: Producción

---

## 📋 Tabla de Contenidos

1. [Baseline del Proyecto](#baseline-del-proyecto)
2. [Arquitectura General](#arquitectura-general)
3. [Estructura de Carpetas](#estructura-de-carpetas)
4. [Componentes Principales](#componentes-principales)
5. [Pipeline de Procesamiento](#pipeline-de-procesamiento)
6. [Convenciones de Organización](#convenciones-de-organización)
7. [Cambios Realizados en esta Limpieza](#cambios-realizados-en-esta-limpieza)

---

## Baseline del Proyecto

### Estado Estable

Este baseline representa la versión **1.0.0** de ActionMiner Lite después de:

1. ✅ Generación de 29 variantes LLM para aumentación de datos
2. ✅ Re-entrenamiento del modelo con dataset mejorado (589 oraciones)
3. ✅ Implementación de 5 estrategias LLM-asistidas
4. ✅ Limpieza y organización completa del repositorio

### Métricas del Baseline

```
Clasificación:
  - F1 Score (test):      0.926
  - Precision:            0.90
  - Recall:               0.96
  - Dataset:              589 oraciones

Extracción:
  - Responsable EM:       0.455 (+14% vs v0.1)
  - Fecha EM:             0.316 (+187% vs v0.1)

Performance:
  - Latencia:             ~12ms/oración
  - Throughput:           ~80 oraciones/seg
```

### Componentes Estables

- **Modelo de clasificación**: `distiluse-base-multilingual-cased-v2` + LogReg
- **NER**: `mrm8488/bert-spanish-cased-finetuned-ner`
- **Parsing de fechas**: `dateparser` con reglas customizadas
- **App**: Streamlit con upload PDF/TXT y export CSV

---

## Arquitectura General

### Diagrama de Alto Nivel

```
┌─────────────────────────────────────────────────────────────┐
│                     ACTIONMINER LITE                        │
│                  Sistema de Detección de Tareas             │
└─────────────────────────────────────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
         ┌──────▼──────┐              ┌──────▼──────┐
         │  Frontend   │              │  Processing │
         │  (Streamlit)│              │  Pipeline   │
         └──────┬──────┘              └──────┬──────┘
                │                             │
                │                      ┌──────┴──────┐
                │                      │             │
                │               ┌──────▼──────┐ ┌───▼────┐
                │               │ Classifier  │ │Extract │
                │               │ (F1 0.926)  │ │(NER+RE)│
                │               └─────────────┘ └────────┘
                │                      │
                └──────────────────────┴──────────────────┐
                                                          │
                                                   ┌──────▼──────┐
                                                   │   Output    │
                                                   │  (CSV/JSON) │
                                                   └─────────────┘
```

### Stack Tecnológico

```
┌─────────────────────────────────────────────────────────────┐
│ CAPA                    │ TECNOLOGÍAS                       │
├─────────────────────────┼───────────────────────────────────┤
│ Frontend                │ Streamlit 1.x                     │
│ ML/NLP                  │ sentence-transformers, sklearn    │
│ NER                     │ transformers (BERT español)       │
│ Parsing                 │ dateparser, regex                 │
│ PDF                     │ pdfplumber                        │
│ Testing                 │ pytest                            │
│ LLM (opcional)          │ anthropic (Claude)                │
└─────────────────────────┴───────────────────────────────────┘
```

---

## Estructura de Carpetas

### Árbol del Proyecto

```
DeL-Proyecto/
│
├── app/                          # Aplicación web
│   └── streamlit_app.py          # App principal Streamlit
│
├── src/                          # Código fuente principal
│   ├── __init__.py
│   │
│   ├── Core Processing:
│   ├── preprocess.py             # Limpieza de texto
│   ├── sentence_split.py         # Segmentación en oraciones
│   ├── featurize.py              # Generación de embeddings
│   ├── postprocess.py            # Normalización de outputs
│   │
│   ├── Classification:
│   ├── train_classifier.py       # Entrenamiento
│   ├── infer_classifier.py       # Inferencia (PRODUCCIÓN)
│   ├── infer_classifier_with_threshold.py  # Con umbral custom
│   │
│   ├── Extraction:
│   ├── ner_extract.py            # Extracción de PERSON (responsable)
│   ├── date_extract.py           # Extracción y normalización de fechas
│   │
│   ├── IO:
│   ├── io_pdf.py                 # Lectura de PDFs
│   │
│   ├── Evaluation:
│   ├── evaluate.py               # Evaluación en test set
│   ├── evaluate_baseline.py      # Evaluación del baseline
│   │
│   ├── experiments/              # Experimentos de ML
│   │   ├── exp01_embeddings_logreg.py     # Baseline (PRODUCCIÓN)
│   │   ├── exp02_bert_finetuning.py       # BERT fine-tuning
│   │   ├── exp03_ensemble.py              # Ensemble
│   │   └── compare_all.py                 # Comparación de modelos
│   │
│   ├── analysis/                 # Análisis y debugging
│   │   ├── analyze_real_errors.py         # Análisis de errores
│   │   └── error_analysis.py              # Error analysis general
│   │
│   ├── llm_augmentation/         # Mejoras con LLM (opcional)
│   │   ├── __init__.py
│   │   ├── generate_difficult_data.py     # Generación de datos
│   │   ├── disambiguate_person.py         # Desambiguación de PERSON
│   │   ├── normalize_dates.py             # Normalización de fechas
│   │   ├── generate_tests.py              # Generación de tests
│   │   ├── enhanced_pipeline.py           # Pipeline con LLM
│   │   └── README.md                      # Doc técnica LLM
│   │
│   └── scripts/                  # Scripts de utilidades
│       ├── integrate_llm_variants.py      # Integrar variantes LLM
│       ├── integrate_tricky_negatives.py  # Integrar negativos
│       └── tune_threshold_improved.py     # Ajuste de umbral
│
├── data/                         # Datos del proyecto
│   ├── annotations/              # Datos anotados
│   │   ├── all_annotations.jsonl          # Dataset completo (589)
│   │   ├── tricky_negatives.jsonl         # Negativos difíciles
│   │   └── llm_generated_variants.jsonl   # Variantes LLM (29)
│   │
│   ├── splits/                   # Train/dev/test
│   │   ├── train.jsonl           # 408 oraciones
│   │   ├── dev.jsonl             # 86 oraciones
│   │   └── test.jsonl            # 95 oraciones
│   │
│   └── scripts/                  # Scripts de datos
│       ├── create_splits.py               # Crear splits
│       ├── generate_tricky_negatives.py   # Generar negativos
│       └── merge_datasets.py              # Merge datasets
│
├── models/                       # Modelos entrenados
│   ├── best_baseline/            # Modelo en producción
│   │   ├── classifier.pkl                 # LogReg
│   │   ├── encoder.pkl                    # SentenceTransformer
│   │   └── threshold.txt                  # Umbral (0.65)
│   │
│   └── exp01_embeddings_logreg/  # Experimentos
│
├── tests/                        # Tests automatizados
│   ├── __init__.py
│   └── unit/                     # Unit tests
│       ├── test_classifier.py
│       ├── test_preprocess.py
│       ├── test_sentence_split.py
│       └── test_date_extract.py
│
├── scripts/                      # Scripts de utilidad
│   ├── test_pipeline.py          # Test end-to-end
│   ├── evaluate_model.py         # Evaluación rápida
│   └── test_document.txt         # Documento de prueba
│
├── eval/                         # Resultados de evaluación
│   ├── test_results_v2.json      # Resultados actuales
│   ├── threshold_tuning_improved.json
│   └── real_data_error_analysis.json
│
├── docs/                         # Documentación
│   ├── CLAUDE.md                 # Plan técnico original
│   ├── INSTRUCCIONES_USO.md      # Cómo usar la app
│   ├── PROYECTO_COMPLETADO.md    # Reporte final
│   ├── MEJORAS_SOBREAJUSTE.md    # Corrección overfitting
│   ├── ESTRATEGIAS_LLM_IMPLEMENTADAS.md
│   ├── RESULTADOS_MEJORA.md
│   ├── RESUMEN_MEJORAS_LLM.md
│   │
│   └── archive/                  # Docs antiguos
│       ├── PROJECT.md
│       ├── RESUMEN_PROYECTO.md
│       └── reports.md
│
├── README.md                     # Documentación principal
├── ARCHITECTURE.md               # Este archivo
├── requirements.txt              # Dependencias Python
└── run_app.sh                    # Script de inicio
```

---

## Componentes Principales

### 1. Core Processing (`src/`)

#### `preprocess.py`
- **Responsabilidad**: Limpieza de texto
- **Funciones**: `clean_text(text) -> str`
- **Transformaciones**: Normalización de espacios, eliminación de footers

#### `sentence_split.py`
- **Responsabilidad**: Segmentación en oraciones
- **Funciones**: `split_sentences(text) -> List[str]`
- **Método**: Regex + reglas para español

#### `featurize.py`
- **Responsabilidad**: Generación de embeddings
- **Modelo**: `sentence-transformers/distiluse-base-multilingual-cased-v2`
- **Output**: Vectores de 512 dimensiones

### 2. Classification (`src/`)

#### `infer_classifier.py` ⭐ PRODUCCIÓN
- **Responsabilidad**: Clasificación TAREA/NO_TAREA
- **Clase**: `SentenceTaskClassifier`
- **Método**: `predict_sentence(text) -> (bool, float)`
- **Modelo**: Embeddings + LogisticRegression
- **Umbral**: 0.65 (configurable en `models/best_baseline/threshold.txt`)

### 3. Extraction (`src/`)

#### `ner_extract.py`
- **Responsabilidad**: Extracción de responsable
- **Función**: `extract_person_responsable(text) -> str`
- **Modelo**: `mrm8488/bert-spanish-cased-finetuned-ner`
- **Fallback**: "pendiente de asignar"

#### `date_extract.py`
- **Responsabilidad**: Extracción y normalización de fechas
- **Función**: `extract_date_iso(text, base_date) -> str`
- **Método**: Regex + dateparser
- **Output**: Formato ISO (YYYY-MM-DD)

### 4. App (`app/`)

#### `streamlit_app.py`
- **Responsabilidad**: Interfaz web
- **Features**:
  - Upload PDF/TXT
  - Procesamiento en tiempo real
  - Visualización de resultados
  - Export a CSV
- **URL**: http://localhost:8501 (por defecto)

---

## Pipeline de Procesamiento

### Flujo Completo

```
┌─────────────┐
│   INPUT     │  PDF, TXT o texto directo
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ PDF Extract │  pdfplumber (si es PDF)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Preprocess  │  clean_text()
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Split     │  split_sentences()
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────┐
│      PARA CADA ORACIÓN           │
│  ┌────────────────────────────┐  │
│  │  1. Featurize (embeddings) │  │
│  └────────┬───────────────────┘  │
│           │                       │
│           ▼                       │
│  ┌────────────────────────────┐  │
│  │  2. Classify               │  │
│  │     (TAREA / NO_TAREA)     │  │
│  └────────┬───────────────────┘  │
│           │                       │
│           ├─ NO_TAREA → Skip     │
│           │                       │
│           └─ TAREA ─┐            │
│                      │            │
│           ┌──────────▼─────────┐ │
│           │  3. Extract        │ │
│           │    - Responsable   │ │
│           │    - Fecha         │ │
│           └────────┬───────────┘ │
│                    │              │
│           ┌────────▼───────────┐ │
│           │  4. Postprocess    │ │
│           │    - Normalize     │ │
│           └────────────────────┘ │
└──────────────────────────────────┘
       │
       ▼
┌─────────────┐
│   OUTPUT    │  JSON / CSV
└─────────────┘
```

### Código de Ejemplo

```python
# Pipeline completo
from pathlib import Path
import sys
sys.path.insert(0, "src")

from io_pdf import pdf_to_text
from preprocess import clean_text
from sentence_split import split_sentences
from infer_classifier import SentenceTaskClassifier
from ner_extract import extract_person_responsable
from date_extract import extract_date_iso

# 1. Extracción (si es PDF)
text = pdf_to_text("documento.pdf")

# 2. Preprocesamiento
cleaned = clean_text(text)

# 3. Segmentación
sentences = split_sentences(cleaned)

# 4. Clasificación y extracción
clf = SentenceTaskClassifier(Path("models/best_baseline"))

for sent in sentences:
    is_task, score = clf.predict_sentence(sent)

    if is_task:
        responsable = extract_person_responsable(sent)
        fecha = extract_date_iso(sent, base_date="2025-11-09")

        print(f"TAREA: {sent}")
        print(f"  Responsable: {responsable}")
        print(f"  Fecha: {fecha}")
```

---

## Convenciones de Organización

### Dónde Poner Nuevos Archivos

#### Código de Producción
- **Módulos core**: `src/nombre_modulo.py`
- **Experimentos**: `src/experiments/exp##_descripcion.py`
- **Análisis**: `src/analysis/nombre_analisis.py`
- **Mejoras LLM**: `src/llm_augmentation/nombre_mejora.py`
- **Scripts de datos**: `src/scripts/nombre_script.py`

#### Datos
- **Anotaciones**: `data/annotations/nombre_dataset.jsonl`
- **Splits**: `data/splits/{train,dev,test}.jsonl`
- **Scripts de datos**: `data/scripts/nombre_script.py`

#### Modelos
- **Modelo en producción**: `models/best_baseline/`
- **Experimentos**: `models/exp##_nombre/`

#### Documentación
- **Docs principales**: `docs/NOMBRE_DOCUMENTO.md`
- **Docs antiguos**: `docs/archive/`
- **README técnicos**: En la carpeta del módulo (ej: `src/llm_augmentation/README.md`)

#### Tests
- **Unit tests**: `tests/unit/test_nombre.py`
- **Integration tests**: `tests/integration/test_nombre.py`
- **Tests LLM-generados**: `tests/test_llm_generated.py`

#### Scripts de Utilidad
- **Scripts standalone**: `scripts/nombre_script.py`
- **Documentos de prueba**: `scripts/nombre_documento.txt`

### Nombres a Evitar

❌ **NO crear archivos con estos nombres**:
- `test.py`, `prueba.py`, `temp.py`, `tmp.py`
- `scratch.py`, `draft.py`, `borrador.py`
- `notas.md`, `resumen.md`, `anotaciones.md`
- `viejo_*.py`, `old_*.py`, `backup_*.py`

✅ **SÍ usar nombres descriptivos**:
- `test_pipeline_complete.py` (si es un test formal)
- `analyze_model_errors.py` (análisis específico)
- `exp04_advanced_features.py` (experimento numerado)

### Convención de Imports

```python
# Estructura recomendada de imports en archivos dentro de src/

# 1. Standard library
import json
import sys
from pathlib import Path
from typing import List, Dict

# 2. Third-party
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# 3. Local (dentro de src/)
from preprocess import clean_text
from infer_classifier import SentenceTaskClassifier
```

### Convención de Estructura de Archivos Python

```python
"""
Docstring del módulo: explicación breve de qué hace
"""

# Imports

# Constantes globales (UPPER_CASE)
DEFAULT_THRESHOLD = 0.65

# Clases
class MiClase:
    pass

# Funciones
def mi_funcion():
    pass

# Main (si aplica)
if __name__ == "__main__":
    main()
```

---

## Cambios Realizados en esta Limpieza

### 📦 Reorganización de Archivos

#### Movidos a `scripts/`
- ✅ `test_pipeline.py` → `scripts/test_pipeline.py`
- ✅ `evaluate_model.py` → `scripts/evaluate_model.py`
- ✅ `test_document.txt` → `scripts/test_document.txt`

**Razón**: Scripts de utilidad que no son parte del código core de producción.

#### Movidos a `docs/`
- ✅ `CLAUDE.md` → `docs/CLAUDE.md`
- ✅ `INSTRUCCIONES_USO.md` → `docs/INSTRUCCIONES_USO.md`
- ✅ `MEJORAS_SOBREAJUSTE.md` → `docs/MEJORAS_SOBREAJUSTE.md`
- ✅ `ESTRATEGIAS_LLM_IMPLEMENTADAS.md` → `docs/ESTRATEGIAS_LLM_IMPLEMENTADAS.md`
- ✅ `RESULTADOS_MEJORA.md` → `docs/RESULTADOS_MEJORA.md`
- ✅ `RESUMEN_MEJORAS_LLM.md` → `docs/RESUMEN_MEJORAS_LLM.md`
- ✅ `PROYECTO_COMPLETADO.md` → `docs/PROYECTO_COMPLETADO.md`

**Razón**: Centralizar documentación en carpeta dedicada.

#### Archivados en `docs/archive/`
- ✅ `PROJECT.md` → `docs/archive/PROJECT.md`
- ✅ `RESUMEN_PROYECTO.md` → `docs/archive/RESUMEN_PROYECTO.md`
- ✅ `eval/reports.md` → `docs/archive/reports.md`

**Razón**: Documentos antiguos o duplicados que no son necesarios en la raíz.

### ⚙️ Actualizaciones de Código

#### `scripts/test_pipeline.py`
- ✅ Actualizado `sys.path.insert(0, ...)` de `parent` a `parent.parent`
- ✅ Actualizado `base_dir` de `parent` a `parent.parent`

#### `scripts/evaluate_model.py`
- ✅ Actualizado `sys.path.insert(0, ...)` de `parent` a `parent.parent`
- ✅ Actualizado `base_dir` de `parent` a `parent.parent`

### 📁 Nuevas Carpetas Creadas

- ✅ `docs/` - Documentación principal
- ✅ `docs/archive/` - Documentación antigua
- ✅ `scripts/` - Scripts de utilidad
- ✅ `scripts/archive/` - Scripts antiguos (vacía por ahora)

### 🗑️ Archivos Eliminados

**Ningún archivo fue eliminado**. Todos los archivos fueron movidos a ubicaciones apropiadas para preservar el historial del proyecto.

### ✅ Archivos Mantenidos en Raíz

Los siguientes archivos permanecen en la raíz por ser esenciales o convencionales:

- ✅ `README.md` - Documentación principal (convención)
- ✅ `ARCHITECTURE.md` - Este archivo (nuevo)
- ✅ `requirements.txt` - Dependencias (convención Python)
- ✅ `run_app.sh` - Script de inicio rápido
- ✅ `.gitignore` - Configuración Git

---

## Verificación de Funcionamiento

### Tests Básicos

```bash
# 1. Test del pipeline completo
python scripts/test_pipeline.py

# 2. Test de la app
streamlit run app/streamlit_app.py

# 3. Unit tests
pytest tests/ -v

# 4. Evaluación en test set
python scripts/evaluate_model.py
```

### Comandos Útiles

```bash
# Ver estructura del proyecto
tree -L 2 -I 'venv|__pycache__|.git|.pytest_cache|models/exp*'

# Correr evaluación completa
python src/evaluate.py data/splits/test.jsonl models/best_baseline

# Generar nuevas variantes LLM (requiere API key)
export ANTHROPIC_API_KEY='tu-key'
python src/llm_augmentation/generate_difficult_data.py
```

---

## Dependencias del Proyecto

### Principales

```
sentence-transformers>=2.0.0    # Embeddings
scikit-learn>=1.0.0             # Clasificación
transformers>=4.20.0            # NER
pdfplumber>=0.7.0               # PDF parsing
dateparser>=1.1.0               # Fecha parsing
streamlit>=1.20.0               # App web
```

### Opcionales

```
anthropic>=0.7.0                # Para mejoras LLM
pytest>=7.0.0                   # Para tests
```

Ver `requirements.txt` para lista completa.

---

## Próximos Pasos Sugeridos

### Para Desarrollo

1. **Mejorar Extracción**:
   - Activar modo enhanced con LLM para casos difíciles
   - Fine-tune NER específico para responsables
   - Mejorar parsing de fechas de proyecto

2. **Nuevos Experimentos**:
   - `exp02_bert_finetuning.py` → F1 > 0.98
   - `exp03_ensemble.py` → Combinar modelos
   - Data augmentation con back-translation

3. **Optimización**:
   - Quantization INT8 del modelo
   - ONNX export para inferencia rápida
   - Batch processing optimizado

### Para Documentación

1. **Agregar**:
   - `docs/API.md` - Documentación de API interna
   - `docs/DEPLOYMENT.md` - Guía de deployment
   - `docs/CONTRIBUTING.md` - Guía para contribuir

---

## Contacto y Mantenimiento

**Proyecto**: ActionMiner Lite
**Versión**: 1.0.0
**Última actualización**: 2025-11-09
**Mantenedor**: Deep Learning Team 2025

---

**Fin del documento ARCHITECTURE.md**
