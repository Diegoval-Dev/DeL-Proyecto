import io
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime

from src.io_pdf import pdf_to_text
from src.preprocess import clean_text
from src.sentence_split import split_sentences
from src.infer_classifier import SentenceTaskClassifier
from src.ner_extract import extract_person_responsable
from src.date_extract import extract_date_iso
from src.postprocess import normalize_row

st.set_page_config(page_title="ActionMiner Lite", layout="wide")

st.title("ActionMiner Lite — MVP")
st.caption("Detecta oraciones TAREA y extrae Responsable (PERSON) y Fecha.")

uploaded = st.file_uploader("Carga un PDF o pega texto en el área de abajo", type=["pdf", "txt"], accept_multiple_files=False)
txt_area = st.text_area("Texto (opcional si no subes archivo)", height=180)

base_date = st.date_input("Fecha base para fechas relativas (dateparser)", value=datetime.today()).strftime("%Y-%m-%d")
doc_id = st.text_input("document_id (para el CSV)", value="demo")

if st.button("Procesar"):
    if uploaded is not None and uploaded.name.lower().endswith(".pdf"):
        content = pdf_to_text(uploaded)
    elif uploaded is not None and uploaded.name.lower().endswith(".txt"):
        content = uploaded.read().decode("utf-8", errors="ignore")
    else:
        content = txt_area

    content = content or ""
    text = clean_text(content)
    sents = split_sentences(text)

    clf = SentenceTaskClassifier(
        model_dir=Path("models")
    )

    rows = []
    for i, s in enumerate(sents):
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

    df = pd.DataFrame(rows)
    st.subheader("Resultados")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Exportar CSV (UTF-8)", data=csv, file_name=f"{doc_id}_actionminer.csv", mime="text/csv")
