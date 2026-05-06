"""DeepSeek embedding model via its OpenAI-compatible API.

Registered as embedder_type="deepseek" in run.py so it works with the
tau2-bench EmbeddingIndexer / EmbeddingEncoder pipeline without modifying
tau2-bench source code.
"""

import os
import time
from typing import List, Optional

import numpy as np
from openai import APIConnectionError, APIError, OpenAI, RateLimitError

from tau2.knowledge.embedders.base import BaseEmbedder

_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
_DEFAULT_MODEL = "text-embedding-v3"


class DeepSeekEmbedder(BaseEmbedder):
    def __init__(self, model: str = _DEFAULT_MODEL, api_key: Optional[str] = None):
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url=_DEEPSEEK_BASE_URL,
        )

    def embed(self, texts: List[str], max_retries: int = 3) -> np.ndarray:
        if not texts:
            raise ValueError("No texts to embed.")

        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model,
                    encoding_format="float",
                )
                return np.array([item.embedding for item in response.data])
            except (APIError, APIConnectionError, RateLimitError) as e:
                last_exc = e
                time.sleep(2 ** attempt)

        raise RuntimeError(
            f"DeepSeek embedding failed after {max_retries} retries. "
            f"Model: {self.model}. Last error: {last_exc}"
        )

    def get_name(self) -> str:
        return f"deepseek_{self.model}"
