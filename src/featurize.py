from sentence_transformers import SentenceTransformer
import numpy as np

def load_encoder(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    return SentenceTransformer(model_name)

def encode_sentences(encoder, texts):
    return encoder.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)

def encode_one(encoder, text: str) -> np.ndarray:
    return encode_sentences(encoder, [text])[0]
