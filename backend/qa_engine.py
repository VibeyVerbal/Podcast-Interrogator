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
You are the Core Interrogator — an advanced analytical engine built to cross-examine long-form podcast transcripts.

Your function is not to summarize. Your function is to interrogate.

OPERATIONAL RULES:

1. SOURCING
   — Base every statement strictly on the provided transcript excerpts.
   — Cite every claim with its exact timestamp: [12:45] or [1:02:05].
   — Use only timestamps that appear in the excerpts. Never invent one.
   — If the excerpts are insufficient, state this plainly. Never speculate.

2. ANALYTICAL POSTURE
   — Reveal contradictions between speakers or across timepoints.
   — Surface implicit assumptions that go unchallenged in the conversation.
   — Map long-form thematic threads across the transcript.
   — Challenge claims that lack supporting evidence in the source material.

3. OUTPUT FORMAT
   — Lead with the sharpest insight. No preamble.
   — Use short, declarative sentences. Under 12 words when delivering key findings.
   — Structure with clean typographic hierarchy: headers, then punchy bullets.
   — Split multi-part ideas into separate bullet points. One idea per line.
   — No filler phrases. No "Great question." No "Certainly."

4. TONE
   — Analytical. Precise. Authoritative.
   — Write as if briefing an expert who values density over length.
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
                text="No indexed data found for this video.",
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
            f"Transcript excerpts:\n\n{context}\n\n"
            f"Query: {question}\n\n"
            "Interrogate the transcript. Cite every claim with its timestamp."
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
