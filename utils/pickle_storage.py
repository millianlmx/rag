import pickle
import os
import numpy as np
from typing import List, Any

class PickleStorage:
    def __init__(self, filename):
        self.filename = filename
        self.data = self._load_or_initialize()

    def _load_or_initialize(self):
        """Load existing data or initialize empty structure"""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'rb') as file:
                    data = pickle.load(file)
                    print(f"Loaded {len(data.get('documents', []))} chunks and vectors from {self.filename}")
                    return data
            except (EOFError, pickle.UnpicklingError):
                print(f"Could not load {self.filename}, initializing empty storage")
        
        return {
            'ids': [],
            'documents': [],
            'embeddings': [],
            'metadatas': []
        }

    def store_vectors(self, ids: List[str], embeddings: List[list], metadatas: List[dict], documents: List[str]):
        """
        Store vectors (embeddings) with associated ids, metadata, and original text chunks.
        """
        # Add new data to existing data
        self.data['ids'].extend(ids)
        self.data['documents'].extend(documents)
        self.data['embeddings'].extend(embeddings)
        self.data['metadatas'].extend(metadatas)
        
        # Save to file
        with open(self.filename, 'wb') as file:
            pickle.dump(self.data, file)
        
        print(f"Saved {len(ids)} new chunks. Total: {len(self.data['documents'])} chunks in {self.filename}")

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    def similarity_search(self, query_vector: list, k: int = 5) -> List[Any]:
        """
        Perform a similarity search and return top k chunks (documents and metadata).
        """
        if not self.data['embeddings']:
            print("No embeddings found in storage")
            return []
        
        # Calculate similarities
        similarities = []
        for i, embedding in enumerate(self.data['embeddings']):
            similarity = self._cosine_similarity(query_vector, embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity (highest first) and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]
        
        # Return results in the expected format
        results = []
        for idx, similarity in top_k:
            results.append({
                "document": self.data['documents'][idx],
                "metadata": self.data['metadatas'][idx],
                "distance": 1 - similarity  # Convert similarity to distance
            })
        
        return results
