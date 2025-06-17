# Chainlit interface for file drop (deposit zone)
import chainlit as cl
from chainlit.types import AskFileResponse
from utils.chromadb_storage import ChromaDBStorage
from utils.doc_parser import pipeline_parser
from utils.llama_cpp_call import LlamaCpp
from typing import List, Union

from utils.pickle_storage import PickleStorage

# Initialize the ChromaDB storage
# storage = ChromaDBStorage()
storage = PickleStorage("knowledge_base.pkl")  # Use PickleStorage for simplicity
llama_cpp = LlamaCpp()

async def process_files(files: Union[List[cl.File], List[AskFileResponse]], init_context = False) -> bool:
    if files:
        if init_context:
            await cl.Message(content=f"Received {len(files)} files. Processing...").send()
        for f in files:
            chunks = pipeline_parser(f)
            if not chunks:
                continue
            # Generate unique ids for each chunk
            ids = [f"{f.id}_{i}" for i in range(len(chunks))]
            documents = [chunk["chunk"] for chunk in chunks]
            embeddings = [(await llama_cpp.embed([chunk["chunk"]]))[0] for chunk in chunks]
            metadatas = [{"file_name": f.name, "chunk_index": chunk["id"], "file_path": f.path} for i, chunk in enumerate(chunks)]
            storage.store_vectors(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
        if init_context:
            await cl.Message(content="Files processed and stored in the knowledge base.").send()
    return True

@cl.on_chat_start
async def on_start(): # Initializes the knowledge base and sends a welcome message
    files = await cl.AskFileMessage(
        content="Please upload files using the deposit zone below.",
        accept=[".pdf", ".pptx", ".docx"],
        max_size_mb=500,
        max_files=3
    ).send()

    await process_files(files, init_context=True)


@cl.on_message
async def main(message: cl.Message):
    query = message.content
    query_vector = (await llama_cpp.embed(query))[0]

    files = list(filter(lambda x: isinstance(x, cl.File), message.elements))

    if files and len(files) > 0:
        fileschunks = [chunk for f in files for chunk in pipeline_parser(f)]

        query = f"""{query}

Provided Document:
{''.join([chunk["chunk"] for chunk in fileschunks]) if fileschunks else "No document content provided."}
"""
        print(f"Query with provided document: {query}")

    # get all relevant hunks from the knowledge base
    results = storage.similarity_search(query_vector, k=5)

    if files:
        await process_files(files)

    # format the results into a readable string like `knowledge base: [chunkContent1, chunkContent2, ...]` in order to send it to the LLM
    context = "\n".join([f"Document: {result['document']}\nMetadata: {result['metadata']}" for result in results])
    # call the LLM with the query and context
    response = await llama_cpp.chat(
        messages=[
            {"role": "system", "content": "Tu es un assistant utile qui répond aux questions en utilisant les documents fournis. Utilise en piorité le document `Provided Document` et ensuite les documents de la base de connaissances."},
            {"role": "user", "content": f"Knowledge base: {context}\n\nQuery: {query}"}
        ]
    )

    # Get unique file names and paths from the results
    unique_files = {result["metadata"]["file_name"]: result["metadata"]["file_path"] for result in results}

    elements = [
        cl.File(
            name=file_name,
            path=file_path,
        ) for file_name, file_path in unique_files.items()
    ]

    # send the response back to the user
    await cl.Message(content=response, elements=elements).send()