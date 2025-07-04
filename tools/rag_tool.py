from typing import List, Dict, Any
from pathlib import Path
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.doc_parser import pipeline_parser
from utils.model_caller import ModelCaller
from utils.pickle_storage import PickleStorage


class RAGTool:
    """
    A tool for Retrieval-Augmented Generation that can process documents,
    store embeddings, and perform similarity searches.
    """
    
    def __init__(self, knowledge_base_path: str = "knowledge_base.pkl", model_caller: ModelCaller = None):
        """
        Initialize the RAG tool.
        
        Args:
            knowledge_base_path: Path to the pickle file for storing the knowledge base
            embedding_model: Name of the sentence transformer model for embeddings
        """
        self.storage = PickleStorage(knowledge_base_path)
        self.model_caller = model_caller or ModelCaller()
        # Get all candidate files
        all_files = list(filter(lambda p: p.endswith((".pdf", ".docx", ".pptx")), map(os.path.join, ["docs"], os.listdir("docs"))))
        # Get already processed file paths from storage
        already_processed = set()
        stats = self.get_knowledge_base_stats()
        for f in stats.get("files", {}).values():
            if "path" in f:
                already_processed.add(f["path"])
                print(f"Already processed: {f['path']}")
        # Only process files not already in storage
        self.document_paths = [f for f in all_files if f not in already_processed]
        if self.document_paths:
            self.process_documents(self.document_paths, verbose=True)
        
    def process_documents(self, file_paths: List[str], verbose: bool = True) -> bool:
        """
        Process documents and add them to the knowledge base.
        
        Args:
            file_paths: List of file paths to process
            verbose: Whether to print processing information
            
        Returns:
            bool: True if processing was successful
        """
        if not file_paths:
            if verbose:
                print("No files provided for processing.")
            return False
            
        if verbose:
            print(f"Processing {len(file_paths)} files...")
            
        processed_count = 0
        
        for file_path in file_paths:
            try:
                # Determine MIME type based on file extension
                file_extension = Path(file_path).suffix.lower()
                mime_type = None
                if file_extension == '.pdf':
                    mime_type = "application/pdf"
                elif file_extension == '.docx':
                    mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                elif file_extension == '.pptx':
                    mime_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                
                # Create a mock file object similar to Chainlit's structure
                file_obj = {
                    'path': file_path,
                    'name': Path(file_path).name,
                    'id': Path(file_path).stem,
                    'type': mime_type  # Use proper MIME type instead of file extension
                }
                
                chunks = pipeline_parser(file_obj)
                if not chunks:
                    if verbose:
                        print(f"No chunks extracted from {file_path}")
                    continue
                    
                # Generate unique IDs for each chunk
                ids = [f"{file_obj['id']}_{i}" for i in range(len(chunks))]
                documents = [chunk["chunk"] for chunk in chunks]
                
                # Generate embeddings for all chunks
                embeddings = []
                for chunk in chunks:
                    embedding = self.model_caller.embed(chunk["chunk"])
                    embeddings.append(embedding)
                
                # Prepare metadata
                metadatas = [
                    {
                        "file_name": file_obj["name"],
                        "chunk_index": chunk["id"],
                        "file_path": file_path
                    } 
                    for i, chunk in enumerate(chunks)
                ]
                
                # Store in knowledge base
                self.storage.store_vectors(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents
                )
                
                processed_count += 1
                if verbose:
                    print(f"Processed {file_path}: {len(chunks)} chunks")
                    
            except Exception as e:
                if verbose:
                    print(f"Error processing {file_path}: {str(e)}")
                continue
                
        if verbose:
            print(f"Successfully processed {processed_count} files.")
            
        return processed_count > 0
    
    async def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform similarity search in the knowledge base.
        
        Args:
            query: The search query
            k: Number of top results to return
            
        Returns:
            List of dictionaries containing document, metadata, and distance
        """
        # Generate embedding for the query
        query_vector = self.model_caller.embed(query)
        
        # Perform similarity search
        results = self.storage.similarity_search(query_vector, k=k)
        
        return results
    
    async def query_with_context(self, 
                                query: str,
                                k: int = 5,
                                attachments: List[Dict[str, Any]] = None,
                                system_prompt: str = None,
                                chat_history: list = None) -> str:
        """
        Query the knowledge base and generate a response using the LLM.
        
        Args:
            query: The user's query
            k: Number of top similar documents to retrieve
            system_prompt: Custom system prompt for the LLM
            
        Returns:
            The LLM's response
        """
        # Format the results into context
        context = "\n".join([
            f"Document: {attachment['document']}\nMetadata: {attachment['metadata']}" 
            for attachment in attachments
        ])
        # Use default system prompt if none provided
        if system_prompt is None:
            system_prompt = (
                "Tu es un assistant utile qui répond aux questions en utilisant "
                "les documents fournis. Utilise en priorité les documents fournis "
                "et ensuite les documents de la base de connaissances."
            )
        # Inject chat history if provided
        messages = []
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": f"Knowledge base: {context}\n\nQuery: {query}"})
        # Call the LLM
        response = await self.model_caller.chat(
            messages=messages
        )
        return response
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current knowledge base.
        
        Returns:
            Dictionary containing statistics
        """
        data = self.storage.data
        
        # Get unique file information
        unique_files = {}
        for metadata in data['metadatas']:
            file_name = metadata.get('file_name', 'unknown')
            if file_name not in unique_files:
                unique_files[file_name] = {
                    'path': metadata.get('file_path', 'unknown'),
                    'chunk_count': 0
                }
            unique_files[file_name]['chunk_count'] += 1
        
        return {
            'total_documents': len(data['documents']),
            'total_embeddings': len(data['embeddings']),
            'unique_files': len(unique_files),
            'files': unique_files
        }
    
    def clear_knowledge_base(self) -> bool:
        """
        Clear the entire knowledge base.
        
        Returns:
            bool: True if cleared successfully
        """
        try:
            self.storage.data = {
                'ids': [],
                'documents': [],
                'embeddings': [],
                'metadatas': []
            }
            # Save empty data
            import pickle
            with open(self.storage.filename, 'wb') as file:
                pickle.dump(self.storage.data, file)
            return True
        except Exception as e:
            print(f"Error clearing knowledge base: {str(e)}")
            return False
