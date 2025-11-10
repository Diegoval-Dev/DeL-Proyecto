import sys
import io
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from io_pdf import pdf_to_text
from preprocess import clean_text
from sentence_split import split_sentences
from infer_classifier import SentenceTaskClassifier
from ner_extract import extract_person_responsable
from date_extract import extract_date_iso
from postprocess import normalize_row

st.set_page_config(page_title="ActionMiner Lite", layout="wide")

st.title("🎯 ActionMiner Lite — Detección de Tareas")
st.caption("Detecta oraciones TAREA y extrae Responsable (PERSON) y Fecha usando NLP.")

# Sidebar con configuración
with st.sidebar:
    st.header("⚙️ Configuración")

    # Selector de modelo (SOLO LOS 3 FINALES)
    model_option = st.selectbox(
        "🎯 Modelo de Clasificación:",
        [
            "🥇 Baseline (Embeddings) - F1: 0.9863 [MEJOR]",
            "🥈 BERT Español Mejorado - F1: 0.9783",
            "🥉 BERT Multilingüe Mejorado - F1: 0.9783"
        ],
        index=0,
        help="Los 3 mejores modelos del proyecto"
    )

    # Mapeo de opciones a rutas
    model_map = {
        "🥇 Baseline (Embeddings) - F1: 0.9863 [MEJOR]": ("best_baseline", "embeddings"),
        "🥈 BERT Español Mejorado - F1: 0.9783": ("exp02_bert_improved/dccuchile_bert-base-spanish-wwm-cased/best_model", "bert"),
        "🥉 BERT Multilingüe Mejorado - F1: 0.9783": ("exp02_bert_improved/bert-base-multilingual-cased/best_model", "bert")
    }

    selected_model_path, selected_model_type = model_map[model_option]

    st.markdown("---")
    st.header("ℹ️ Información")
    st.markdown("""
    **Pipeline del sistema:**
    1. 📝 Extracción de texto (PDF/TXT)
    2. 🔍 Clasificación (modelo seleccionado)
    3. 👤 NER para responsables (BERT)
    4. 📅 Extracción de fechas (dateparser)

    **Los 3 modelos finales:**
    - 🥇 **Baseline:** Mejor F1 (0.9863)
    - 🥈 **BERT Español:** Balance P/R (0.9783)
    - 🥉 **BERT Multilingüe:** Backup (0.9783)

    **Dataset entrenamiento:**
    - 408 oraciones originales
    - 86 oraciones validación
    - 500+ en total con test set
    """)

st.markdown("### 📄 Entrada de Datos")
col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader(
        "Carga un PDF o archivo de texto",
        type=["pdf", "txt"],
        accept_multiple_files=False
    )
    txt_area = st.text_area(
        "O pega tu texto aquí:",
        height=200,
        placeholder="Ejemplo: Juan debe enviar el informe antes del viernes. Se discutió el presupuesto del proyecto."
    )

with col2:
    base_date = st.date_input(
        "📅 Fecha base para fechas relativas",
        value=datetime.today(),
        help="Usada para interpretar 'mañana', 'próxima semana', etc."
    ).strftime("%Y-%m-%d")

    doc_id = st.text_input(
        "🏷️ ID del documento",
        value="demo",
        help="Identificador para el archivo CSV exportado"
    )

if st.button("🚀 Procesar Documento", type="primary"):
    # Cargar contenido
    if uploaded is not None and uploaded.name.lower().endswith(".pdf"):
        with st.spinner("Extrayendo texto del PDF..."):
            content = pdf_to_text(uploaded)
    elif uploaded is not None and uploaded.name.lower().endswith(".txt"):
        content = uploaded.read().decode("utf-8", errors="ignore")
    else:
        content = txt_area

    if not content or len(content.strip()) < 10:
        st.warning("⚠️ Por favor ingresa o carga un documento con texto.")
    else:
        # Preprocesar
        with st.spinner("Preprocesando texto..."):
            text = clean_text(content)
            sents = split_sentences(text)

        st.info(f"📊 Se encontraron **{len(sents)}** oraciones en el documento.")

        # Cargar clasificador según selección
        with st.spinner(f"Cargando modelo..."):
            base_dir = Path(__file__).parent.parent
            clf = SentenceTaskClassifier(
                model_dir=base_dir / "models" / selected_model_path
            )

        # Procesar oraciones
        rows = []
        progress_bar = st.progress(0)

        for i, s in enumerate(sents):
            progress_bar.progress((i + 1) / len(sents))

            is_task, score = clf.predict_sentence(s)

            if is_task:
                responsable = extract_person_responsable(s)
                fecha_iso = extract_date_iso(s, base_date)
            else:
                responsable = ""
                fecha_iso = ""

            row = {
                "document_id": doc_id,
                "sent_id": i,
                "oracion": s,
                "es_tarea": bool(is_task),
                "score": float(score),
                "responsable": responsable,
                "fecha_iso": fecha_iso,
            }
            rows.append(normalize_row(row))

        progress_bar.empty()

        # Crear DataFrame
        df = pd.DataFrame(rows)
        df_tareas = df[df["es_tarea"] == True].copy()

        # Mostrar resultados
        st.markdown("---")
        st.subheader("📋 Resultados del Análisis")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Oraciones", len(df))
        with col2:
            st.metric("Tareas Detectadas", len(df_tareas))
        with col3:
            porcentaje = (len(df_tareas) / len(df) * 100) if len(df) > 0 else 0
            st.metric("% Tareas", f"{porcentaje:.1f}%")

        # Tabs para diferentes vistas
        tab1, tab2 = st.tabs(["🎯 Solo Tareas", "📑 Todo el Documento"])

        with tab1:
            if len(df_tareas) > 0:
                st.dataframe(
                    df_tareas[["sent_id", "oracion", "responsable", "fecha_iso", "score"]],
                    width='stretch',
                    hide_index=True
                )
            else:
                st.info("ℹ️ No se detectaron tareas en el documento.")

        with tab2:
            st.dataframe(
                df[["sent_id", "oracion", "es_tarea", "responsable", "fecha_iso", "score"]],
                width='stretch',
                hide_index=True
            )

        # Exportar CSV
        st.markdown("---")
        st.subheader("💾 Exportar Resultados")

        col1, col2 = st.columns(2)

        with col1:
            csv_all = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Descargar TODO (CSV)",
                data=csv_all,
                file_name=f"{doc_id}_completo.csv",
                mime="text/csv"
            )

        with col2:
            if len(df_tareas) > 0:
                csv_tasks = df_tareas.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Descargar SOLO TAREAS (CSV)",
                    data=csv_tasks,
                    file_name=f"{doc_id}_tareas.csv",
                    mime="text/csv"
                )

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
    ActionMiner Lite - Proyecto Deep Learning 2025 |
    Modelo: {model_option.split(' - ')[0]} |
    Dataset: 408-749 oraciones (original + augmented)
    </div>
    """,
    unsafe_allow_html=True
)
