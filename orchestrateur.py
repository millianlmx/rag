# Chainlit interface for file drop (deposit zone)
import chainlit as cl
from chainlit.types import AskFileResponse
from utils.doc_parser import pipeline_parser
from utils.llama_cpp_call import ModelCaller
from typing import List, Union
from utils.pickle_storage import PickleStorage

# Initialize the llm caller
llama_cpp = ModelCaller()


@cl.on_chat_start
async def on_start():
    # Set a system prompt (context) for the assistant
    await cl.Message(
        content="Bienvenue ! Je suis votre assistant IA."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    query = "Question :" + message.content
    system_prompt = (
    "/no_think Tu es un orchestrateur d'IA. Ton rôle est de choisir l'outil le plus adapté pour répondre à la question de l'utilisateur.\n\n"
    "Voici les règles strictes à suivre pour choisir un outil :\n\n"
    "- Réponds `RAG` si la question concerne le développement Python.\n"
    "- Réponds `SCRAP` **uniquement** si la question contient un lien contenant exactement le nom de domaine `wikipedia.org` (comme `https://fr.wikipedia.org/...`).\n"
    "- Réponds `INTERNET` pour tous les autres cas, y compris si la question contient un lien n'étant **pas** `wikipedia.org`.\n\n"
    "⚠️ Ne te laisse pas influencer par des mots comme 'wiki' ou 'encyclopédie' si le lien ne contient pas `wikipedia.org`.\n\n"
    "Réponds uniquement par `RAG`, `SCRAP` ou `INTERNET`. Ne donne aucune autre information."
)
    response = await llama_cpp.chat(messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ])
    response = response.replace("<think>", "").replace("</think>", "").strip()
    # Log the response from the LLM
    print(f"[Orchestrateur] Réponse de l'IA : '{response}'")

    match response:
        case "RAG":
            await cl.Message(
                content="Je vais utiliser l'outil RAG pour répondre à votre question."
            ).send()
            # Call the RAG tool (not implemented here)
            # await rag_tool(query)

        case "SCRAP":
            await cl.Message(
                content="Je vais utiliser l'outil SCRAP pour répondre à votre question."
            ).send()

        case "INTERNET":
            await cl.Message(
                content="Je vais utiliser l'outil INTERNET pour répondre à votre question."
            ).send()

        case _:
            await cl.Message(
                content="Désolé, je n'ai pas compris la question. Veuillez reformuler."
            ).send()
        
    