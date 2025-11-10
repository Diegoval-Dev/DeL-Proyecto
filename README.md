# 🎯 ActionMiner Lite - Detección de Tareas con NLP

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Sistema de procesamiento de lenguaje natural para detectar tareas en documentos en español.**

---

## 🌟 Características

- ✅ **Clasificación de Tareas**: Detecta oraciones que contienen tareas con F1=0.9863
- 👤 **Extracción de Responsables**: Identifica personas usando NER en español
- 📅 **Normalización de Fechas**: Convierte fechas absolutas y relativas a formato ISO
- 🎨 **Interfaz Web**: Aplicación Streamlit profesional y fácil de usar
- 📊 **Exportación CSV**: Descarga resultados en formato estructurado
- 📄 **Soporte PDF/TXT**: Procesa múltiples formatos de entrada

## 🚀 Inicio Rápido

### Instalación

```bash
# Clonar el repositorio
git clone [tu-repo]
cd DeL-Proyecto

# Instalar dependencias
pip install -r requirements.txt
```

### Uso

#### Opción 1: Script de Lanzamiento (Recomendado)

```bash
./run_app.sh
```

#### Opción 2: Comando Directo

```bash
streamlit run app/streamlit_app.py
```

#### Opción 3: Verificar Pipeline

```bash
python test_pipeline.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📖 Ejemplo de Uso

### Entrada

```text
Juan debe enviar el informe antes del viernes 15 de noviembre.
Se discutió el presupuesto del proyecto.
María coordinará la reunión con el equipo técnico el próximo martes.
```

### Salida (CSV)

| sent_id | oracion | es_tarea | score | responsable | fecha_iso |
|---------|---------|----------|-------|-------------|-----------|
| 0 | Juan debe enviar el informe... | TRUE | 0.989 | Juan | 2025-11-15 |
| 1 | Se discutió el presupuesto... | FALSE | 0.024 | - | - |
| 2 | María coordinará la reunión... | TRUE | 0.979 | María | 2025-11-12 |

## 🏗️ Arquitectura

```
📄 Entrada (PDF/TXT/Texto)
    ↓
🧹 Preprocesamiento
    ↓
✂️ Segmentación en Oraciones
    ↓
🤖 Clasificación (Embeddings + LogReg)
    ↓
SI es TAREA →
    👤 NER (Responsable)
    📅 Extracción de Fecha
    ↓
💾 Exportación CSV
```

## 📊 Rendimiento del Modelo

### Métricas en Test Set

| Métrica | Valor |
|---------|-------|
| **F1 Score** | 0.9863 |
| **Precision** | 0.9744 |
| **Recall** | 1.0000 |
| **Accuracy** | 0.9867 |

### Dataset

- **Total**: 500 oraciones etiquetadas
- **Balance**: 53% TAREA / 47% NO_TAREA
- **Splits**: 70% train / 15% dev / 15% test

### Modelos Entrenados

| Experimento | Modelo | F1 (dev) | Estado |
|-------------|--------|----------|--------|
| Embeddings + LogReg | Spanish Embeddings | 1.0000 | ✅ Mejor |
| BERT Fine-tuning | Multilingual BERT | 1.0000 | ✅ |
| Ensemble | Soft Voting | 1.0000 | ✅ |

## 📁 Estructura del Proyecto

```
DeL-Proyecto/
├── app/
│   └── streamlit_app.py          # Aplicación web principal
├── src/
│   ├── preprocess.py             # Limpieza de texto
│   ├── sentence_split.py         # Segmentación
│   ├── infer_classifier.py       # Clasificación
│   ├── ner_extract.py            # Extracción de responsables
│   ├── date_extract.py           # Extracción de fechas
│   └── experiments/              # Scripts de experimentación
├── data/
│   ├── annotations/              # Dataset etiquetado (500 oraciones)
│   └── splits/                   # Train/dev/test
├── models/
│   └── best_baseline/            # Mejor modelo (F1=0.9863)
├── tests/
│   └── unit/                     # 17 tests unitarios
├── eval/                         # Resultados y visualizaciones
├── INSTRUCCIONES_USO.md          # Guía detallada de usuario
├── RESUMEN_PROYECTO.md           # Resumen ejecutivo completo
└── test_pipeline.py              # Script de verificación
```

## 🧪 Testing

Ejecutar tests unitarios:

```bash
pytest tests/unit/ -v
```

Verificar pipeline completo:

```bash
python test_pipeline.py
```

## 📚 Documentación

- **[INSTRUCCIONES_USO.md](INSTRUCCIONES_USO.md)**: Guía completa de usuario
- **[RESUMEN_PROYECTO.md](RESUMEN_PROYECTO.md)**: Resumen ejecutivo y resultados
- **[CLAUDE.md](CLAUDE.md)**: Especificaciones técnicas detalladas

## 🛠️ Tecnologías Utilizadas

- **Python 3.10+**: Lenguaje principal
- **Transformers (Hugging Face)**: Modelos BERT y NER
- **Sentence-Transformers**: Embeddings de oraciones
- **Scikit-learn**: Clasificación y grid search
- **Streamlit**: Interfaz web
- **dateparser**: Normalización de fechas
- **pdfplumber**: Extracción de texto de PDFs
- **pytest**: Testing

## 🎓 Casos de Uso

### 1. Análisis de Actas de Reunión

Extrae automáticamente:
- Tareas asignadas a cada persona
- Fechas límite de entrega
- Compromisos adquiridos

### 2. Gestión de Emails Corporativos

Identifica:
- Solicitudes de acción
- Responsables de seguimiento
- Plazos de respuesta

### 3. Procesamiento de Documentos Legales/Administrativos

Detecta:
- Obligaciones contractuales
- Fechas de vencimiento
- Partes responsables

## 🔍 Funcionalidades Avanzadas

### Clasificación Inteligente

- Modelo entrenado con 500 oraciones reales
- Grid search con 16 combinaciones de hiperparámetros
- Calibración de umbral por F1 score

### Extracción de Responsables

- NER BERT fine-tuned en español
- Vinculación contextual con verbos de acción
- Detección de proximidad responsable-verbo

### Normalización de Fechas

- Fechas absolutas: "15/11/2025", "15 de noviembre de 2025"
- Fechas relativas: "mañana", "próximo martes", "esta semana"
- Salida en formato ISO (YYYY-MM-DD)

## 🚧 Limitaciones Conocidas

- ❗ Solo procesa documentos con texto embebido (no OCR)
- ❗ Optimizado para español de España/Latinoamérica
- ❗ NER puede fallar con nombres poco comunes
- ❗ Fechas ambiguas dependen de la fecha base configurada

## 🤝 Contribuciones

Para reportar bugs o sugerir mejoras:

1. Revisar documentación existente
2. Ejecutar `test_pipeline.py` para reproducir
3. Incluir ejemplos específicos del problema

## 📝 Changelog

### v1.0.0 (Noviembre 2025)

- ✅ Sistema completo funcional
- ✅ 3 experimentos de ML completados
- ✅ Interfaz Streamlit profesional
- ✅ 17 tests unitarios pasando
- ✅ F1 = 0.9863 en test set
- ✅ Documentación completa

## 📄 Licencia

Este proyecto fue desarrollado como parte del curso de Deep Learning en la Universidad del Valle de Guatemala.

## 👥 Autores

**Proyecto**: ActionMiner Lite
**Curso**: Deep Learning y Sistemas Inteligentes
**Institución**: Universidad del Valle de Guatemala
**Año**: 2025

---

## 🏃 Comandos Rápidos

```bash
# Instalar
pip install -r requirements.txt

# Ejecutar app
./run_app.sh
# o
streamlit run app/streamlit_app.py

# Probar pipeline
python test_pipeline.py

# Tests
pytest tests/unit/ -v

# Experimentos
python src/experiments/exp01_embeddings_logreg.py
python src/experiments/compare_all.py
```

---

**Estado**: ✅ Completado y Funcional

**Última actualización**: Noviembre 2025

Para más información, consulta [INSTRUCCIONES_USO.md](INSTRUCCIONES_USO.md) o [RESUMEN_PROYECTO.md](RESUMEN_PROYECTO.md)
