# High-level ChromaDB vector storage and retrieval
import chromadb
from chromadb.config import Settings
from typing import List, Any

class ChromaDBStorage:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.Client(Settings(persist_directory=persist_directory, is_persistent=True))
        self.collection = self.client.get_or_create_collection("documents")

    def store_vectors(self, ids: List[str], embeddings: List[list], metadatas: List[dict], documents: List[str]):
        """
        Store vectors (embeddings) with associated ids, metadata, and original text chunks.
        Forces persistence to disk after adding.
        """
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    def similarity_search(self, query_vector: list, k: int = 5) -> List[Any]:
        """
        Perform a similarity search and return top k chunks (documents and metadata).
        """
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        # Return a list of dicts for each result
        return [
            {
                # 'ids' is not included in the result, so we skip it
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            }
            for i in range(len(results["documents"][0]))
        ]
