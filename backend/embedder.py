"""
Step 3 of the pipeline: Embed.

Converts text chunks (and later, user questions) into vectors using
Google's Gemini embedding model via the google-genai SDK.
"""

import time
from typing import List

from google import genai
from google.genai import types


class EmbeddingError(Exception):
    pass


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str, output_dimensionality: int = 768):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.output_dimensionality = output_dimensionality

    def _embed(self, texts: List[str], task_type: str, max_retries: int = 4) -> List[List[float]]:
        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self.output_dimensionality,
        )
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=texts,
                    config=config,
                )
                return [embedding.values for embedding in response.embeddings]
            except Exception as e:  # transient API/network errors -> backoff and retry
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
        raise EmbeddingError(f"Embedding request failed after {max_retries} attempts: {last_error}") from last_error

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embed transcript chunks for storage. Uses the RETRIEVAL_DOCUMENT
        task type, which Gemini optimizes differently than query embeddings
        for better retrieval quality."""
        embeddings: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            embeddings.extend(self._embed(batch, task_type="RETRIEVAL_DOCUMENT"))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a user's question. Uses the RETRIEVAL_QUERY task type."""
        return self._embed([text], task_type="RETRIEVAL_QUERY")[0]
