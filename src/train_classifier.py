import json, joblib, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.model_selection import train_test_split

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main(train_path="data/annotations/train.jsonl", model_dir="models"):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    data = list(load_jsonl(train_path))
    X_text = [d["text"] for d in data]
    y = np.array([1 if d["label"] == "TAREA" else 0 for d in data])

    enc = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    X = enc.encode(X_text, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)

    Xtr, Xdev, ytr, ydev = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)

    scores = clf.predict_proba(Xdev)[:, 1]
    prec, rec, th = precision_recall_curve(ydev, scores)
    f1 = (2 * prec * rec) / (prec + rec + 1e-9)
    best_idx = f1.argmax()
    best_th = th[best_idx]
    print("F1_dev:", float(f1[best_idx]), "Threshold:", float(best_th))

    joblib.dump(clf, model_dir / "classifier.pkl")
    joblib.dump(enc, model_dir / "sentence_encoder.pkl")
    (model_dir / "threshold.txt").write_text(str(best_th), encoding="utf-8")

if __name__ == "__main__":
    main()
