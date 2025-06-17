import pickle
from typing import List, Any

class PickleStorage:
    def __init__(self, filename):
        self.filename = filename

    def save(self, data):
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
        # results = self.collection.query(
        #     query_embeddings=[query_vector],
        #     n_results=k,
        #     include=["documents", "metadatas", "distances"]
        # )
        
        with open(self.filename, 'rb') as file:
            data = pickle.load(file)
            print(f"Loaded {len(data['documents'])} chunks and vectors from {self.filename}")
            
            
            # return data['chunks'], data['vectors']
        
        # Return a list of dicts for each result
        return [
            {
                # 'ids' is not included in the result, so we skip it
                "document": data["documents"][0][i],
                "metadata": data["metadatas"][0][i],
                "distance": data["distances"][0][i]
            }
            for i in range(len(data["documents"][0]))
        ]
