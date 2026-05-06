"""Local sentence-transformers embedder — no API key required.

Registered as embedder_type="local" in run.py. Uses any Hugging Face
sentence-transformers model; defaults to all-MiniLM-L6-v2 (22 MB, 384 dims)
which is fast and adequate for English retrieval tasks.
"""

from typing import List, Optional

import numpy as np

from tau2.knowledge.embedders.base import BaseEmbedder

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class LocalEmbedder(BaseEmbedder):
    def __init__(self, model: str = _DEFAULT_MODEL):
        self.model_name = model
        self._model = None  # lazy-load on first use

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            raise ValueError("No texts to embed.")
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.astype(np.float32)

    def get_name(self) -> str:
        return f"local_{self.model_name}"
