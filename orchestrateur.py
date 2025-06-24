# Chainlit interface for file drop (deposit zone)
import chainlit as cl
from chainlit.types import AskFileResponse
from utils.model_caller import ModelCaller
from typing import List, Dict, Union
from tools.rag_tool import RAGTool
from tools.internet_search_tool import InternetSearchTool
import json 

# Initialize the llm caller
model_caller = ModelCaller()

# Initialize tools
rag_tool = RAGTool(knowledge_base_path="knowledge_base.pkl", model_caller=model_caller)
internet_search_tool = InternetSearchTool(model_caller=model_caller)

# Global chat history: list of dicts {"role": "user"|"assistant", "content": ...}
chat_history = []


@cl.on_chat_start
async def on_start():
    # Set a system prompt (context) for the assistant
    await cl.Message(
        content="Bienvenue ! Je suis votre assistant IA. Je peux répondre à vos questions en utilisant ma base de connaissances ou en recherchant sur Internet.\n\n"
    ).send()


async def handle_rag_query(query: str):
    """Handle queries using the RAG tool"""
    global chat_history
    try:
        # Get knowledge base stats first
        stats = rag_tool.get_knowledge_base_stats()
        
        # Check if the knowledge base is empty
        if stats['total_documents'] == 0:
            await cl.Message(
                content="Ma base de connaissances est vide. Je vais chercher sur Internet pour vous aider."
            ).send()
            return await handle_internet_query(query)
        
        # Get similarity search results to extract source files
        results = await rag_tool.similarity_search(query, k=5)
        
        # Use RAG tool to get response
        # Only keep the last N turns (e.g., 5), and only user/assistant roles
        history_to_inject = chat_history[-10:] if len(chat_history) > 0 else []
        response = await rag_tool.query_with_context(query, k=5, attachments=results, chat_history=history_to_inject)
        
        # Extract unique source files from results
        unique_files = {}
        
        for result in results:
            metadata = result.get('metadata', {})
            file_name = metadata.get('file_name')
            file_path = metadata.get('file_path')
            
            if file_name and file_path:
                unique_files[file_name] = file_path
        
        print(f"[RAG Debug] Unique files found: {list(unique_files.keys())}")
        
        # Create cl.File elements for sources
        elements = []
        for file_name, file_path in unique_files.items():
            try:
                # Check if file exists before creating element
                import os
                
                if os.path.exists(file_path):
                    elements.append(cl.File(name=file_name, path=file_path))
                else:
                    # Try to find the file in temp_uploads directory
                    temp_path = os.path.join(os.getcwd(), "temp_uploads", file_name)
                    if os.path.exists(temp_path):
                        elements.append(cl.File(name=file_name, path=temp_path))
            except Exception as e:
                print(f"[RAG] Error creating file element for {file_name}: {e}")
        
        await cl.Message(content=response, elements=elements).send()
        # Update chat history: add user query and assistant answer
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": response})
        
    except Exception as e:
        print(f"[RAG Tool Error] {str(e)}")
        await cl.Message(
            content="Désolé, j'ai rencontré un problème avec ma base de connaissances. Laissez-moi chercher sur Internet."
        ).send()
        await handle_internet_query(query)


async def handle_internet_query(query: str):
    """Handle queries using the Internet Search tool"""
    global chat_history
    try:
        await cl.Message(content="🔍 Recherche en cours sur Internet...").send()
        
        
        # Use internet search tool to get summarized response
        # Inject chat history (user/assistant turns) before the current query
        history_to_inject = chat_history[-10:] if len(chat_history) > 0 else []
        response = await internet_search_tool.generate_answer_from_web(
            query,
            num_results=5,
            num_extract=3,
            chat_history=history_to_inject
        )
        await cl.Message(content=response).send()
        # Update chat history: add user query and assistant answer
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": response})
        
    except Exception as e:
        print(f"[Internet Search Error] {str(e)}")
        await cl.Message(
            content="Désolé, je n'ai pas pu effectuer la recherche sur Internet. Veuillez réessayer."
        ).send()


async def handle_wikipedia_scrap(query: str, wikipedia_url: str):
    """Handle Wikipedia content extraction"""
    global chat_history
    try:
        await cl.Message(content="📖 Extraction du contenu Wikipedia en cours...").send()
        
        # Extract content from Wikipedia URL
        content = await internet_search_tool.extract_webpage_text(wikipedia_url, max_chars=5000)
        
        if content['content'] and len(content['content']) > 50:
            # Use the extracted content to answer the query
            system_prompt = (
                "Tu es un assistant utile qui répond aux questions en utilisant "
                "le contenu Wikipedia fourni. Fournis une réponse complète et précise "
                "basée sur les informations extraites."
            )
            
            # Inject chat history (user/assistant turns) before the current query
            history_to_inject = chat_history[-10:] if len(chat_history) > 0 else []
            messages = history_to_inject + [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Contenu Wikipedia:\n{content['content']}\n\nQuestion: {query}"}
            ]
            response = await model_caller.chat(messages=messages)
            await cl.Message(content=f"{response}\n\n**Source:** {wikipedia_url}").send()
            # Update chat history: add user query and assistant answer
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": response})
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


@cl.on_message
async def main(message: cl.Message):

    query = message.content.strip()
    original_query = query
    
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
    
    response = await model_caller.chat(messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": orchestrator_query}
    ])

    # Get rid of think tags (on some LLMs use these to indicate thinking)
    response: Dict[str, Union[str, List[str]]] = json.loads(response.replace("<think>", "").replace("</think>", "").strip().replace("```json\n", "").replace("\n```", ""))

    # Enhance variables from orchestrator response
    tool: str = response.get("tool", "").upper()
    urls: List[str] = response.get("urls", [])

    # Switch based on the tool selected by the orchestrator
    match(tool):
        case "RAG":
            await cl.Message(
            content="🧠 Je vais consulter ma base de connaissances pour répondre à votre question."
            ).send()
            await handle_rag_query(original_query)

        case "INTERNET":
            await cl.Message(
            content="🌐 Je vais rechercher sur Internet pour vous donner une réponse à jour."
            ).send()
            await handle_internet_query(original_query)

        case "SCRAPING":
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

        case "LLM" | _:
            # We consider the LLM tool as default
            # Inject chat history (user/assistant turns) before the current query
            history_to_inject = chat_history[-10:] if len(chat_history) > 0 else []
            messages = [
                {"role": "system", "content": "Tu es un assistant utile qui répond aux questions de manière conversationnelle. N'utilisa pas d'émojies."},
            ] + history_to_inject + [
                {"role": "user", "content": original_query}
            ]
            response = await model_caller.chat(messages=messages)
            await cl.Message(content=response).send()
            # Update chat history: add user query and assistant answer
            chat_history.append({"role": "user", "content": original_query})
            chat_history.append({"role": "assistant", "content": response})

# Add cleanup function for the internet search tool and temporary files
@cl.on_chat_end
async def on_chat_end():
    """Cleanup when chat session ends"""
    global chat_history
    try:
        await internet_search_tool.close()
    except:
        pass
    # Clean up chat history
    chat_history.clear()
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

