# Chainlit interface for file drop (deposit zone)
import chainlit as cl
from chainlit.types import AskFileResponse
from utils.model_caller import ModelCaller
from typing import List, Dict, Union
from tools.rag_tool import RAGTool
from tools.internet_search_tool import InternetSearchTool
import json 

# Initialize the llm caller
llama_cpp = ModelCaller()

# Initialize tools
rag_tool = RAGTool(knowledge_base_path="knowledge_base.pkl")
internet_search_tool = InternetSearchTool()


@cl.on_chat_start
async def on_start():
    # Set a system prompt (context) for the assistant
    await cl.Message(
        content="Bienvenue ! Je suis votre assistant IA. Je peux répondre à vos questions en utilisant ma base de connaissances ou en recherchant sur Internet.\n\n📄 **Pour télécharger des documents :**\n- Glissez-déposez vos fichiers dans le chat\n- Ou tapez `/upload` pour sélectionner des fichiers\n\n🔍 **Formats supportés :** PDF, DOCX, PPTX"
    ).send()


async def handle_rag_query(query: str):
    """Handle queries using the RAG tool"""
    try:
        # Get knowledge base stats first
        stats = rag_tool.get_knowledge_base_stats()
        
        if stats['total_documents'] == 0:
            await cl.Message(
                content="Ma base de connaissances est vide. Je vais chercher sur Internet pour vous aider."
            ).send()
            return await handle_internet_query(query)
        
        # Get similarity search results to extract source files
        results = await rag_tool.similarity_search(query, k=5)
        
        # Use RAG tool to get response
        response = await rag_tool.query_with_context(query, k=5)
        
        # Extract unique source files from results
        unique_files = {}
        print(f"[RAG Debug] Found {len(results)} similarity results")
        
        for result in results:
            metadata = result.get('metadata', {})
            file_name = metadata.get('file_name')
            file_path = metadata.get('file_path')
            
            print(f"[RAG Debug] Result metadata: file_name={file_name}, file_path={file_path}")
            
            if file_name and file_path:
                unique_files[file_name] = file_path
        
        print(f"[RAG Debug] Unique files found: {list(unique_files.keys())}")
        
        # Create cl.File elements for sources
        elements = []
        for file_name, file_path in unique_files.items():
            try:
                # Check if file exists before creating element
                import os
                print(f"[RAG Debug] Checking file: {file_path}, exists: {os.path.exists(file_path)}")
                
                if os.path.exists(file_path):
                    elements.append(cl.File(name=file_name, path=file_path))
                    print(f"[RAG Debug] Added file element: {file_name}")
                else:
                    print(f"[RAG] Source file not found: {file_path}")
                    # Try to find the file in temp_uploads directory
                    temp_path = os.path.join(os.getcwd(), "temp_uploads", file_name)
                    if os.path.exists(temp_path):
                        elements.append(cl.File(name=file_name, path=temp_path))
                        print(f"[RAG Debug] Found file in temp_uploads: {temp_path}")
            except Exception as e:
                print(f"[RAG] Error creating file element for {file_name}: {e}")
        
        print(f"[RAG Debug] Created {len(elements)} file elements")
        await cl.Message(content=response, elements=elements).send()
        
    except Exception as e:
        print(f"[RAG Tool Error] {str(e)}")
        await cl.Message(
            content="Désolé, j'ai rencontré un problème avec ma base de connaissances. Laissez-moi chercher sur Internet."
        ).send()
        await handle_internet_query(query)


async def handle_internet_query(query: str):
    """Handle queries using the Internet Search tool"""
    try:
        await cl.Message(content="🔍 Recherche en cours sur Internet...").send()
        
        
        # Use internet search tool to get summarized response
        response = await internet_search_tool.generate_answer_from_web(
            query,
            num_results=5,
            num_extract=3
        )
        
        await cl.Message(content=response).send()
        
    except Exception as e:
        print(f"[Internet Search Error] {str(e)}")
        await cl.Message(
            content="Désolé, je n'ai pas pu effectuer la recherche sur Internet. Veuillez réessayer."
        ).send()


async def handle_wikipedia_scrap(query: str, wikipedia_url: str):
    """Handle Wikipedia content extraction"""
    try:
        await cl.Message(content="📖 Extraction du contenu Wikipedia en cours...").send()
        
        # Extract content from Wikipedia URL
        content = await internet_search_tool.extract_content(wikipedia_url, max_chars=5000)
        
        if content['content'] and len(content['content']) > 50:
            # Use the extracted content to answer the query
            system_prompt = (
                "Tu es un assistant utile qui répond aux questions en utilisant "
                "le contenu Wikipedia fourni. Fournis une réponse complète et précise "
                "basée sur les informations extraites."
            )
            
            response = await llama_cpp.chat(messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Contenu Wikipedia:\n{content['content']}\n\nQuestion: {query}"}
            ])
            
            await cl.Message(content=f"{response}\n\n**Source:** {wikipedia_url}").send()
        else:
            await cl.Message(
                content="Je n'ai pas pu extraire le contenu de cette page Wikipedia. Laissez-moi faire une recherche Internet."
            ).send()
            await handle_internet_query(query)
            
    except Exception as e:
        print(f"[Wikipedia Scraping Error] {str(e)}")
        await cl.Message(
            content="Désolé, je n'ai pas pu extraire le contenu de Wikipedia. Laissez-moi faire une recherche Internet."
        ).send()
        await handle_internet_query(query)

# Add file processing capability through message handling
async def process_uploaded_files(files: List[AskFileResponse]):
    """Process uploaded files and add them to the knowledge base"""
    if not files:
        return False
    
    await cl.Message(content="📄 Traitement des fichiers en cours...").send()
    
    try:
        file_paths = []
        import os
        
        # Create a temporary directory if it doesn't exist
        temp_dir = os.path.join(os.getcwd(), "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        
        for file in files:
            debug_file_object(file, "Upload")
            
            # Handle different file types
            content = None
            
            # AskFileResponse objects have 'content' as bytes or 'path' attribute
            if hasattr(file, 'content') and file.content is not None:
                content = file.content
                print(f"[File Upload] Using direct content, size: {len(content)} bytes")
            elif hasattr(file, 'path') and file.path:
                # Read from path
                try:
                    with open(file.path, 'rb') as f:
                        content = f.read()
                    print(f"[File Upload] Read from path {file.path}, size: {len(content)} bytes")
                except Exception as e:
                    print(f"[File Upload] Error reading from path {file.path}: {e}")
                    continue
            else:
                print(f"[File Upload] No content or path found for file {file.name}")
                continue
            
            if not content or len(content) == 0:
                print(f"[File Upload] Warning: File {file.name} has no content")
                await cl.Message(
                    content=f"❌ Le fichier {file.name} semble être vide."
                ).send()
                continue
            
            # Save the uploaded file temporarily with a safe filename
            safe_filename = file.name.replace(" ", "_").replace("(", "").replace(")", "").replace("—", "-")
            temp_path = os.path.join(temp_dir, safe_filename)
                
            with open(temp_path, "wb") as f:
                bytes_written = f.write(content)
                print(f"[File Upload] Wrote {bytes_written} bytes to {temp_path}")
            
            # Verify the file was written correctly
            if os.path.exists(temp_path):
                file_size = os.path.getsize(temp_path)
                print(f"[File Upload] File created successfully: {temp_path}, size: {file_size} bytes")
                if file_size == 0:
                    print(f"[File Upload] ERROR: File {temp_path} is empty!")
                    continue
            else:
                print(f"[File Upload] ERROR: File {temp_path} was not created!")
                continue
                
            file_paths.append(temp_path)
            print(f"[File Upload] Added to processing queue: {temp_path}")

        # Process files with RAG tool
        if not file_paths:
            await cl.Message(
                content="❌ Aucun fichier valide trouvé pour le traitement."
            ).send()
            return False
            
        success = await rag_tool.process_documents(file_paths, verbose=True)
        
        if success:
            stats = rag_tool.get_knowledge_base_stats()
            await cl.Message(
                content=f"✅ Fichiers traités avec succès ! Ma base de connaissances contient maintenant {stats['total_documents']} documents."
            ).send()
        else:
            await cl.Message(
                content="❌ Erreur lors du traitement des fichiers. Veuillez réessayer."
            ).send()
            
        # Keep files for source references - don't clean up immediately
        # Files will be cleaned up when the chat session ends
        if success:
            print(f"[File Upload] Keeping {len(file_paths)} files for source references")
        else:
            # Only clean up if processing failed
            for path in file_paths:
                try:
                    os.remove(path)
                    print(f"[File Upload] Cleaned up failed file: {path}")
                except Exception as cleanup_error:
                    print(f"[File Upload] Could not clean up {path}: {cleanup_error}")
            
            # Remove temp directory if empty
            try:
                os.rmdir(temp_dir)
            except:
                pass  # Directory not empty or other issue, ignore
        
        return success
                
    except Exception as e:
        print(f"[File Upload Error] {str(e)}")
        await cl.Message(
            content="❌ Erreur lors du traitement des fichiers. Veuillez réessayer."
        ).send()
        return False


@cl.on_message
async def main(message: cl.Message):

    query = message.content.strip()
    original_query = query

    
    if None:
        # Extract Wikipedia content directly
        await handle_wikipedia_scrap(query, wikipedia_urls[0])
        return
    
    # Use orchestrator to determine which tool to use
    orchestrator_query = "Question :" + query
    system_prompt = (
        "Tu es un orchestrateur intelligent qui décide quel outil utiliser pour répondre à la question de l'utilisateur.\n"
        "Règles de décision :\n"
        "- RAG : Utilise cet outil quand la question concerne la programmation\n"
        "- SCRAPING : Utilise cet outil quand tu détectes une URL Wikipedia (https://fr.wikipedia.org/wiki/...) dans la question, assure toi de scraper uniquement les pages Wikipedia avec le nom de domaine `fr.wikipedia.org` sinon renvoie INTERNET /!\\ /!\\.\n"
        "- INTERNET : Utilise cet outil quand la question nécessite des connaissances générales récentes ou spécialisées\n"
        "- LLM : Utilise cet outil pour les questions banales ou conversationnelles simples\n\n"
        "Réponds uniquement avec ce format JSON :\n"
        "{\n"
        "    \"tool\": \"RAG\" | \"SCRAPING\" | \"INTERNET\" | \"LLM\",\n"
        "    \"urls\": [\"url1\", \"url2\", ...]\n"
        "}\n\n"
        "Si tu suggères `SCRAPING`, inclus les URLs détectées dans le champ `urls`. Sinon, laisse le champ `urls` vide [].\n"
        "Ne donne aucune autre information que ce JSON."
    )
    
    response = await llama_cpp.chat(messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": orchestrator_query}
    ])

    response: Dict[str, Union[str, List[str]]] = json.loads(response.replace("<think>", "").replace("</think>", "").strip().replace("```json\n", "").replace("\n```", ""))

    tool: str = response.get("tool", "").upper()
    urls: List[str] = response.get("urls", [])

    if "RAG" in tool:
        await cl.Message(
            content="🧠 Je vais consulter ma base de connaissances pour répondre à votre question."
        ).send()
        await handle_rag_query(original_query)

    elif "INTERNET" in tool:
        await cl.Message(
            content="🌐 Je vais rechercher sur Internet pour vous donner une réponse à jour."
        ).send()
        await handle_internet_query(original_query)
    elif "SCRAPING" in tool:
        if urls and len(urls) > 0:
            wikipedia_url = urls[0]
            if "fr.wikipedia.org" in wikipedia_url:
                await handle_wikipedia_scrap(original_query, wikipedia_url)
            else:
                await cl.Message(
                    content="⚠️ L'URL fournie n'est pas une page Wikipedia valide. Je vais faire une recherche Internet à la place."
                ).send()
                await handle_internet_query(original_query)
        else:
            await cl.Message(
                content="⚠️ Aucune URL Wikipedia détectée. Je vais faire une recherche Internet à la place."
            ).send()
            await handle_internet_query(original_query)
    else:
        # We consider the LLM tool as default
        response = await llama_cpp.chat(messages=[
            {"role": "system", "content": "Tu es un assistant utile qui répond aux questions de manière conversationnelle."},
            {"role": "user", "content": original_query}
        ])
        await cl.Message(content=response).send()

# Add cleanup function for the internet search tool and temporary files
@cl.on_chat_end
async def on_chat_end():
    """Cleanup when chat session ends"""
    try:
        await internet_search_tool.close()
    except:
        pass
    
    # Clean up temporary uploaded files
    try:
        import os
        import shutil
        temp_dir = os.path.join(os.getcwd(), "temp_uploads")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"[Cleanup] Removed temporary files directory: {temp_dir}")
    except Exception as e:
        print(f"[Cleanup] Error removing temp files: {e}")

