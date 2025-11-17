import os, math, logging
from typing import List, Tuple
from models.embeddings import EmbeddingModel
import numpy as np
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

class SimpleFAISS:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []  # parallel list of texts

    def add(self, vectors: np.ndarray, texts: List[str]):
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        assert vectors.shape[1] == self.dim, 'Dimension mismatch'
        self.index.add(vectors.astype('float32'))
        self.texts.extend(texts)

    def search(self, query_vector: np.ndarray, k: int = 4) -> List[Tuple[str, float]]:
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        D, I = self.index.search(query_vector.astype('float32'), k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < len(self.texts) and idx != -1:
                results.append((self.texts[idx], float(dist)))
        return results

def build_index_from_documents(doc_texts: List[str], embedder: EmbeddingModel, chunk_size=300, overlap=50):
    all_chunks = []
    for t in doc_texts:
        chunks = chunk_text(t, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)
    if not all_chunks:
        raise ValueError('No document text provided to build index.')
    embs = embedder.embed(all_chunks)
    dim = embs.shape[1]
    index = SimpleFAISS(dim)
    index.add(embs, all_chunks)
    return index

def retrieve_relevant(query: str, index: SimpleFAISS, embedder: EmbeddingModel, k=4):
    q_emb = embedder.embed(query)[0]
    results = index.search(q_emb, k=k)
    return results
