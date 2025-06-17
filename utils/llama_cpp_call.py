# LlamaCpp class for LLM and embedding calls using HTTP requests (OpenAI API compatible endpoints)
import httpx

class LlamaCpp:
    def __init__(self, llm_url: str = "http://127.0.0.1:8080/v1", embedding_url: str = "http://127.0.0.1:8081"):
        self.llm_url = llm_url
        self.embedding_url = embedding_url

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

    async def embed(self, text: str, model="Qwen3-Embedding-0.6B-GGUF", **kwargs):
        """
        Async call to the embedding endpoint with a list of texts.
        text: A single string to embed (will be wrapped in a list for API call)
        model: Must match an embedding model loaded by llama.cpp server.
        Returns a list of embedding vectors.
        """
        url = f"{self.embedding_url}/embeddings"
        payload = {
            "model": model,
            "content": text
        }
        payload.update(kwargs)
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data[0]["embedding"]
