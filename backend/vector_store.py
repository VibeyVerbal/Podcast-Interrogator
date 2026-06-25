"""
Storage layer used by both Step 3 (Embed) and Step 4 (Retrieve).

Each video gets its own ChromaDB collection (named yt_<video_id>), stored
in a local, persistent database on disk. This means once a video has been
indexed, you can chat with it again later without re-fetching or
re-embedding anything.
"""

from typing import List, Optional

import chromadb

from chunker import Chunk

_COLLECTION_PREFIX = "yt_"


class VectorStore:
    def __init__(self, db_path: str):
        self.client = chromadb.PersistentClient(path=db_path)

    @staticmethod
    def _collection_name(video_id: str) -> str:
        return f"{_COLLECTION_PREFIX}{video_id}"

    def has_video(self, video_id: str) -> bool:
        name = self._collection_name(video_id)
        return any(c.name == name for c in self.client.list_collections())

    def _get_or_create(self, video_id: str):
        return self.client.get_or_create_collection(
            name=self._collection_name(video_id),
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        video_id: str,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        video_title: Optional[str] = None,
    ) -> None:
        collection = self._get_or_create(video_id)
        collection.add(
            ids=[f"{video_id}_{c.chunk_id}" for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "video_id": video_id,
                    "video_title": video_title or video_id,
                    "start_time": c.start_time,
                    "end_time": c.end_time,
                }
                for c in chunks
            ],
        )

    def query(self, video_id: str, query_embedding: List[float], n_results: int = 5):
        collection = self._get_or_create(video_id)
        n_results = min(n_results, max(collection.count(), 1))
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    def delete_video(self, video_id: str) -> None:
        try:
            self.client.delete_collection(self._collection_name(video_id))
        except Exception:
            pass  # nothing to delete

    def list_videos(self) -> List[str]:
        return [
            c.name[len(_COLLECTION_PREFIX) :]
            for c in self.client.list_collections()
            if c.name.startswith(_COLLECTION_PREFIX)
        ]
