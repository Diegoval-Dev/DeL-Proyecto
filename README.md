# ActionMiner Lite — README (MVP)

> Proyecto académico — Detección de **oraciones TAREA** en español y extracción de **Responsable** y **Fecha** a partir de texto o PDF.

---

## 1. Resumen

**ActionMiner Lite** identifica **oraciones que representan tareas** en documentos (actas, correos, PDFs) y, para cada tarea, extrae **Responsable** (PERSON) y **Fecha**. El resultado se presenta en una tabla y puede exportarse a **CSV**.

**Alcance del MVP**

* ✓ Detección de oraciones **TAREA** (clasificador fine‑tuneado por el equipo).
* ✓ Extracción de **Responsable** (NER + reglas) y **Fecha** (regex + `dateparser`).
* ✓ Ingesta de **texto** y **PDF** (extracción de texto).
* ✓ **App Streamlit** para demostración + export **CSV**.
* ✗ Fuera de alcance: DECISIÓN/INFO, resúmenes, export `.ics`, fine‑tuning del NER.

---

## 2. Arquitectura (pipeline)

```
[Input texto/PDF]
    └─► Prepro (PDF→texto, limpieza)
        └─► Segmentación a oraciones
            └─► Clasificador oracional (fine‑tune del equipo) → {TAREA | no‑TAREA}
                └─► (solo TAREA) NER (preentrenado) → entidades PERSON/DATE
                    └─► Reglas de vínculo Responsable (PERSON) ↔ verbo‑acción
                    └─► Normalización Fecha (regex + dateparser)
                        └─► Tabla final + export CSV
```

---

## 3. Requisitos

* **Python** ≥ 3.10
* CPU; GPU opcional.

**requirements.txt**

```
transformers>=4.43
torch>=2.2
sentence-transformers>=3.0
scikit-learn>=1.4
dateparser>=1.2
pdfplumber>=0.11
streamlit>=1.36
```

---

## 4. Estructura del repositorio

```
actionminer/
  app/
    streamlit_app.py
  data/
    raw/             # textos/PDF crudos (anonimizados)
    interim/         # textos limpios
    annotations/     # dataset etiquetado (jsonl/csv)
    splits/          # listas de train/dev/test
  models/
    sentence_encoder/   # cache (opcional)
    classifier.pkl      # clasificador TAREA
    threshold.txt       # umbral calibrado por F1
  src/
    io_pdf.py
    preprocess.py
    sentence_split.py
    featurize.py
    train_classifier.py
    infer_classifier.py
    ner_extract.py
    date_extract.py
    postprocess.py
    evaluate.py
  eval/
    reports.md
  README.md
```

---

## 5. Instalación y ejecución rápida

```bash
# 1) Entorno
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Dependencias
pip install -r requirements.txt

# 3) Entrenar clasificador (usa data/annotations/*.jsonl)
python src/train_classifier.py

# 4) Ejecutar app
streamlit run app/streamlit_app.py
```

---

## 6. Dataset y etiquetado

**Entrada de entrenamiento**: oraciones etiquetadas como `TAREA` o `NO_TAREA`. Para algunas `TAREA`, incluyen `responsable_gold` y `fecha_gold` para evaluación de extracción.

**Formato `jsonl` (una oración por línea):**

```json
{"doc_id":"acta_001","sent_id":0,"text":"Juan enviará el informe el jueves.","label":"TAREA","responsable_gold":"Juan","fecha_gold":"2025-10-23"}
{"doc_id":"acta_001","sent_id":1,"text":"Se discutió el presupuesto.","label":"NO_TAREA"}
```

**Criterios de etiquetado**

* `TAREA` si hay verbo de acción atribuible (p. ej., enviar, preparar, entregar, revisar, coordinar, agendar, subir, compartir, firmar, actualizar, configurar, documentar, notificar, resolver, investigar, programar, instalar, comprar, validar, corregir, reportar) y expresa obligación/compromiso (debe, tiene que, se acuerda que X hará Y).
* `NO_TAREA` si solo informa/discute ("se habló", "se presentó").
* Casos límite: “Se debe enviar el informe.” es `TAREA`; el responsable puede quedar “pendiente de asignar”.

**Splits**

* Separar por documento: 70% train / 15% dev / 15% test.

---

## 7. Entrenamiento del clasificador (intervención propia)

**Estrategia**

* **Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
* **Clasificador**: LogisticRegression (o SVM lineal) con validación cruzada.
* **Calibración de umbral**: elegir umbral que maximice F1 en `dev`.

**Salida del entrenamiento**

* `models/classifier.pkl`, `models/sentence_encoder.pkl`, `models/threshold.txt` (umbral en flotante).

---

## 8. Inferencia y post‑proceso

**Inferencia (oraciones)**

* Cargar `classifier.pkl` + `threshold.txt` → `es_tarea: bool`.

**NER + reglas para Responsable**

* NER (español) con agregación de entidades PERSON.
* Reglas: (1) PERSON a la izquierda del verbo de acción, mínima distancia; (2) PERSON a la derecha vinculada por "a/para"; (3) múltiples PERSON → más cercana al verbo; (4) si no hay PERSON → “pendiente de asignar”.

**Fecha**

* Regex para fechas absolutas/relativas + `dateparser.parse(..., languages=['es'], settings={'RELATIVE_BASE': base_dt})`.
* Si no parsea → “sin fecha”.

**Verbos de acción (lista editable)**

```
{"enviar","preparar","entregar","revisar","coordinar","agendar","subir","compartir",
 "firmar","actualizar","configurar","documentar","notificar","resolver","investigar",
 "programar","instalar","comprar","validar","corregir","reportar"}
```

---

## 9. Evaluación

**Tareas del script `src/evaluate.py`**

* **Clasificador**: F1 (macro) en test; precisión/recobrado por clase.
* **Extracción**: Exact‑Match de `responsable` y `fecha` cuando existan etiquetas `*_gold`.
* **Latencia**: tiempo promedio por documento en CPU.

**Criterios objetivo (ajustables)**

* F1(TAREA) ≥ 0.80.
* Responsable Exact‑Match ≥ 0.70 (si hay PERSON).
* Fecha Exact‑Match ≥ 0.70 (si hay fecha).
* Latencia ≤ 2 s por documento (1–2 páginas) en CPU.

---

## 10. App (Streamlit)

**Uso**

1. Cargar texto (textarea) o PDF (uploader).
2. Procesar → segmentación → clasificación TAREA.
3. Para cada TAREA → NER/Fecha → tabla.
4. Descargar resultados como **CSV**.

**CSV de salida**

```
document_id,sent_id,oracion,es_tarea,responsable,fecha_iso
```

---

## 11. Intervención propia y originalidad

* **Dataset propio** (recolección, anonimización y etiquetado por el equipo).
* **Fine‑tuning del clasificador** (embeddings + modelo lineal), con validación cruzada y calibración por F1.
* **Reglas lingüísticas** de vínculo Responsable‑acción y normalización de fechas.
* **Error analysis** y ajustes iterativos (listas de verbos, exclusión de verbos de reporte, mejoras de regex).
* **Ablation**: comparación cero‑shot vs fine‑tune y con/sin reglas.

---

## 12. Ejemplo reproducible

**Entrada**

> “Juan enviará el informe el jueves. Se revisó el presupuesto. Ana debe coordinar la reunión para la próxima semana.”

**Salida (CSV)**

```
document_id,sent_id,oracion,es_tarea,responsable,fecha_iso
demo,0,"Juan enviará el informe el jueves.",true,"Juan","2025-10-23"
demo,1,"Se revisó el presupuesto.",false,,
demo,2,"Ana debe coordinar la reunión para la próxima semana.",true,"Ana","2025-10-24"
```

---

## 13. Solución de problemas (FAQ)

* **Torch no instala con CUDA** → usar versión CPU (por defecto) o instalar `--index-url` oficial de PyTorch.
* **PDF sin texto** → el PDF es imagen; para el MVP usar PDFs con texto embebido o convertir previamente con OCR externo.
* **Fechas relativas ambiguas** → ajustar `RELATIVE_BASE` (p. ej., fecha de la minuta) y documentar política.
* **Nombres compuestos** → unir spans contiguos PERSON en `postprocess.py`.

---

## 14. Roadmap (post‑MVP)

* Export **.ics**, clasificación DECISIÓN/INFO, resumen breve.
* Fine‑tuning ligero del NER en dominio local.
* Mejora de heurísticas (coreferencia básica, priors por remitente/destinatario de correo).

---

## 15. Licencia

MIT 
---

## 16. Autores

* [Diego Valenzuela] — [22309]
* [Gerson Ramirez] — [22281]

Curso/Sección — Universidad — Período.
