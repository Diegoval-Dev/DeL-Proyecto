import json, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, classification_report
from src.infer_classifier import SentenceTaskClassifier

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main(test_path="data/annotations/test.jsonl", model_dir="models"):
    data = list(load_jsonl(test_path))
    texts = [d["text"] for d in data]
    y_true = [1 if d["label"]=="TAREA" else 0 for d in data]

    clf = SentenceTaskClassifier(Path(model_dir))
    y_pred = [int(clf.predict_sentence(t)[0]) for t in texts]

    print("F1 (macro) TAREA/no-TAREA:", f1_score(y_true, y_pred, average="macro"))
    print(classification_report(y_true, y_pred, target_names=["NO_TAREA","TAREA"]))

    df = pd.DataFrame(data)
    if {"responsable_gold","fecha_gold"}.issubset(df.columns):
        from src.ner_extract import extract_person_responsable
        from src.date_extract import extract_date_iso
        resp_pred = []
        date_pred = []
        for t in texts:
            resp_pred.append(extract_person_responsable(t))
            date_pred.append(extract_date_iso(t, base_date=pd.Timestamp.today().date().isoformat()))
        df["responsable_pred"] = resp_pred
        df["fecha_pred"] = date_pred

        mask_has_resp = df["responsable_gold"].fillna("").astype(str).str.len() > 0
        mask_has_date = df["fecha_gold"].fillna("").astype(str).str.len() > 0
        resp_em = (df.loc[mask_has_resp, "responsable_pred"].str.strip().str.lower()
                   == df.loc[mask_has_resp, "responsable_gold"].str.strip().str.lower()).mean() if mask_has_resp.any() else float("nan")
        date_em = (df.loc[mask_has_date, "fecha_pred"].astype(str).str.strip()
                   == df.loc[mask_has_date, "fecha_gold"].astype(str).str.strip()).mean() if mask_has_date.any() else float("nan")
        print(f"Exact-Match Responsable (cuando hay gold): {resp_em:.3f}")
        print(f"Exact-Match Fecha (cuando hay gold): {date_em:.3f}")

if __name__ == "__main__":
    main()
