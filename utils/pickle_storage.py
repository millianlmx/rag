import pickle
from typing import List, Any
from scipy.spatial.distance import cosine

class PickleStorage:
    def __init__(self, filename):
        self.filename = filename

    def store_vectors(self, ids: List[str], embeddings: List[list], metadatas: List[dict], documents: List[str]):
        data = {
            "ids": ids,
            "embeddings": embeddings,
            "metadatas": metadatas,
            "documents": documents
        }

        with open(self.filename, 'wb') as file:
            pickle.dump(data, file)

        print(f"Saved {len(data.chunks)} chunks and vectors to {data.vectors}")
    

    def load(self):
        import pickle
        with open(self.filename, 'rb') as file:
            return pickle.load(file) if file else None
        
    def similarity_search(self, query_vector: list, k: int = 5) -> List[Any]:
        """
        Perform a similarity search and return top k chunks (documents and metadata).
        """
        with open(self.filename, 'rb') as file:
            data = pickle.load(file)
            print(f"Loaded {len(data['documents'])} chunks and vectors from {self.filename}")
            
            
            # return data['chunks'], data['vectors']
        loaded_data = [
            {
                "document": data["documents"][0][i],
                "metadata": data["metadatas"][0][i],
                "distance": 1 - cosine(query_vector, data["embeddings"][0][i])  # Calculate cosine distance
            }
            for i in range(len(data["documents"][0]))
        ]
        
        # Sort data by distance
        loaded_data.sort(key=lambda x: x["distance"], reverse=True)
        
        return loaded_data[:k]  # Return only the top k results
