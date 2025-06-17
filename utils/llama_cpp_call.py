# LlamaCpp class for LLM calls and SentenceTransformers for embeddings
import httpx
from sentence_transformers import SentenceTransformer

class ModelCaller:
    def __init__(self, llm_url: str = "http://127.0.0.1:8080/v1", embedding_model: str = "Lajavaness/sentence-camembert-large"):
        self.llm_url = llm_url
        self.embedding_model = SentenceTransformer(embedding_model)

    async def chat(self, messages, model="gemma-3-4b-it-GGUF", **kwargs):
        """
        Async call to the LLM endpoint with a list of messages (OpenAI format).
        messages: List of dicts, e.g. [{"role": "user", "content": "Hello"}]
        model: Must match a model loaded by llama.cpp server.
        """
        url = f"{self.llm_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages
        }
        payload.update(kwargs)
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code == 400:
                print("[LlamaCpp] 400 Error:", response.text)
                response.raise_for_status()
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def embed(self, text: str, **kwargs):
        """
        Async call using SentenceTransformers for embedding generation.
        text: A single string to embed
        Returns an embedding vector as a list.
        """
        # SentenceTransformers encode method is synchronous, but we wrap it for async compatibility
        import asyncio
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self.embedding_model.encode, text)
        return embedding.tolist()