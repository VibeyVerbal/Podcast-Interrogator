"""
Steps 4 & 5 of the pipeline: Retrieve and Generate.

Turns the user's question into a vector, pulls back the most relevant
transcript chunks for that video, then hands them to Gemini to synthesize
a grounded, timestamp-cited answer.
"""

from dataclasses import dataclass
from typing import List

from google import genai
from google.genai import types

from chunker import format_timestamp
from embedder import GeminiEmbedder
from vector_store import VectorStore

SYSTEM_PROMPT = """\
You are a precise research assistant. You answer questions about a video \
using ONLY the transcript excerpts you are given below — never your own \
outside knowledge of the topic.

Rules:
1. Base every part of your answer strictly on the excerpts provided. If the \
   excerpts don't contain enough to answer, say so plainly rather than \
   guessing or relying on what you already know about the subject.
2. After each claim, cite the timestamp it came from, in square brackets, \
   e.g. [12:45] or [1:02:05]. Use only timestamps that appear in the \
   excerpts — never invent one.
3. Write like you're briefing someone who hasn't watched the video: clear, \
   conversational, and to the point. No need to repeat the excerpts verbatim.
"""


@dataclass
class Citation:
    start_time: float
    timestamp: str
    preview: str


@dataclass
class Answer:
    text: str
    citations: List[Citation]


class QAEngine:
    def __init__(
        self,
        embedder: GeminiEmbedder,
        store: VectorStore,
        api_key: str,
        generation_model: str,
        top_k: int = 5,
    ):
        self.embedder = embedder
        self.store = store
        self.client = genai.Client(api_key=api_key)
        self.generation_model = generation_model
        self.top_k = top_k

    def ask(self, video_id: str, question: str) -> Answer:
        query_embedding = self.embedder.embed_query(question)
        results = self.store.query(video_id, query_embedding, n_results=self.top_k)

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        if not documents:
            return Answer(
                text="This video doesn't seem to be indexed yet, so I have nothing to search.",
                citations=[],
            )

        citations: List[Citation] = []
        context_blocks: List[str] = []
        for doc, meta in zip(documents, metadatas):
            ts = format_timestamp(meta["start_time"])
            context_blocks.append(f"[{ts}] {doc}")
            citations.append(Citation(start_time=meta["start_time"], timestamp=ts, preview=doc[:160]))

        context = "\n\n---\n\n".join(context_blocks)
        prompt = (
            f"Transcript excerpts from the video:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the excerpts above, with timestamp citations."
        )

        response = self.client.models.generate_content(
            model=self.generation_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
            ),
        )

        return Answer(text=response.text or "(empty response)", citations=citations)
