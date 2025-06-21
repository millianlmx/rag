# Chainlit interface for file drop (deposit zone)
import chainlit as cl
from chainlit.types import AskFileResponse
from utils.llama_cpp_call import ModelCaller
from typing import List
from tools.rag_tool import RAGTool
from tools.internet_search_tool import InternetSearchTool
import re

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
        
        # Get search results and extracted content for sources
        search_data = await internet_search_tool.search_and_extract(
            query,
            num_results=5,
            num_extract=3
        )
        
        # Use internet search tool to get summarized response
        response = await internet_search_tool.search_and_summarize(
            query,
            num_results=5,
            num_extract=3
        )
        
        # Add source information to the response
        if search_data.get('search_results'):
            sources_text = "\n\n**Sources:**\n"
            for i, result in enumerate(search_data['search_results'][:3], 1):
                sources_text += f"{i}. [{result['title']}]({result['url']})\n"
            response += sources_text
        
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


# Add explicit file upload handling
async def ask_for_files():
    """Ask user to upload files for processing"""
    files = await cl.AskFileMessage(
        content="Veuillez télécharger vos documents (PDF, DOCX, PPTX) :",
        accept=["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                "application/vnd.openxmlformats-officedocument.presentationml.presentation"],
        max_files=10,
        max_size_mb=20
    ).send()
    
    if files:
        success = await process_uploaded_files(files)
        return success
    return False


@cl.on_message
async def main(message: cl.Message):
    # Handle file uploads first
    if message.elements:
        print(f"[Debug] Found {len(message.elements)} elements in message")
        files = []
        
        for element in message.elements:
            print(f"[Debug] Element type: {type(element)}")
            print(f"[Debug] Element attributes: {dir(element)}")
            
            # Handle Chainlit File elements
            if hasattr(element, 'path') and hasattr(element, 'name'):
                try:
                    import os  # Import os for file operations
                    print(f"[Debug] Reading file from path: {element.path}")
                    print(f"[Debug] Original file exists: {os.path.exists(element.path)}")
                    
                    with open(element.path, 'rb') as f:
                        content = f.read()
                    
                    print(f"[Debug] Read {len(content)} bytes from original file")
                    
                    if len(content) == 0:
                        print(f"[Debug] ERROR: Original file {element.path} is empty!")
                        await cl.Message(
                            content=f"❌ Le fichier {element.name} semble être vide ou corrompu."
                        ).send()
                        continue
                    
                    # Create a mock AskFileResponse-like object with proper attributes
                    # Determine MIME type based on file extension
                    file_extension = element.name.lower().split('.')[-1]
                    mime_type = None
                    if file_extension == 'pdf':
                        mime_type = "application/pdf"
                    elif file_extension == 'docx':
                        mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    elif file_extension == 'pptx':
                        mime_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    
                    file_obj = type('FileObj', (), {
                        'content': content,
                        'name': element.name,
                        'type': mime_type,  # For AskFileResponse compatibility
                        'path': element.path,
                        'id': element.name.split('.')[0]  # Use filename without extension as ID
                    })()
                    files.append(file_obj)
                    print(f"[Debug] Successfully read file: {element.name}, size: {len(content)} bytes, mime: {mime_type}")
                    
                except Exception as e:
                    print(f"[Debug] Could not read file {element.path}: {e}")
                    await cl.Message(
                        content=f"❌ Erreur lors de la lecture du fichier {element.name}: {str(e)}"
                    ).send()
                    
            # Handle other file types (AskFileResponse)
            elif hasattr(element, 'content') and hasattr(element, 'name'):
                if element.content is not None:
                    files.append(element)
                    print(f"[Debug] Found file with content: {element.name}")
                else:
                    print(f"[Debug] File {element.name} has no content")
        
        if files:
            await process_uploaded_files(files)
            return
        else:
            await cl.Message(
                content="❌ Aucun fichier valide détecté. Veuillez réessayer."
            ).send()
            return
    
    query = message.content.strip()
    original_query = query
    
    # Check for special commands
    if query.lower() in ['/upload', '/add_files', '/documents']:
        await ask_for_files()
        return
    
    # Check if the message contains a Wikipedia URL
    wikipedia_pattern = r'https?://[a-zA-Z0-9.-]*\.?wikipedia\.org/[^\s]*'
    wikipedia_urls = re.findall(wikipedia_pattern, query)
    
    if wikipedia_urls:
        # Extract Wikipedia content directly
        await handle_wikipedia_scrap(query, wikipedia_urls[0])
        return
    
    # Use orchestrator to determine which tool to use
    orchestrator_query = "Question :" + query
    system_prompt = (
        "/no_think Tu es un orchestrateur d'IA. Ton rôle est de choisir l'outil le plus adapté pour répondre à la question de l'utilisateur.\n\n"
        "Voici les règles strictes à suivre pour choisir un outil :\n\n"
        "- Réponds `RAG` si la question concerne le développement Python, la programmation, ou des sujets techniques qui pourraient être dans une base de connaissances locale.\n"
        "- Réponds `INTERNET` pour toutes les autres questions, y compris les actualités, les événements récents, les informations générales, ou toute question nécessitant des informations à jour.\n\n"
        "Réponds uniquement par `RAG` ou `INTERNET`. Ne donne aucune autre information."
    )
    
    response = await llama_cpp.chat(messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": orchestrator_query}
    ])
    response = response.replace("<think>", "").replace("</think>", "").strip()
    
    # Log the response from the LLM
    print(f"[Orchestrateur] Réponse de l'IA : '{response}' pour la question : '{query}'")

    if "RAG" in response.upper():
        await cl.Message(
            content="🧠 Je vais consulter ma base de connaissances pour répondre à votre question."
        ).send()
        await handle_rag_query(original_query)
        
    elif "INTERNET" in response.upper():
        await cl.Message(
            content="🌐 Je vais rechercher sur Internet pour vous donner une réponse à jour."
        ).send()
        await handle_internet_query(original_query)

    # To do: implement code execution tool like python, it will make the chatbot able to perform calculations, data analysis, graphs, etc.
        
    else:
        # Default to internet search if unclear
        await cl.Message(
            content="🌐 Je vais rechercher sur Internet pour répondre à votre question."
        ).send()
        await handle_internet_query(original_query)


# Debug function to inspect file objects
def debug_file_object(file_obj, context=""):
    """Debug function to inspect file object attributes"""
    print(f"[Debug {context}] File object type: {type(file_obj)}")
    print(f"[Debug {context}] File name: {getattr(file_obj, 'name', 'NO NAME')}")
    
    # Check common attributes
    attrs_to_check = ['content', 'path', 'id', 'type', 'mime', 'size']
    for attr in attrs_to_check:
        if hasattr(file_obj, attr):
            value = getattr(file_obj, attr)
            if attr == 'content' and value:
                print(f"[Debug {context}] {attr}: {type(value)} with {len(value)} bytes")
            else:
                print(f"[Debug {context}] {attr}: {value}")
        else:
            print(f"[Debug {context}] {attr}: NOT PRESENT")
    
    print(f"[Debug {context}] All attributes: {[a for a in dir(file_obj) if not a.startswith('_')]}")


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

