from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name: str):
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f'Failed to load embedding model {model_name}: {e}')

    def embed(self, texts):
        # texts: list[str] or str
        if isinstance(texts, str):
            texts = [texts]
        embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embs
