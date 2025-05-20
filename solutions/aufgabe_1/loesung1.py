import os
import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

load_dotenv()

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_KEY")
api_version = os.getenv("AZURE_OPENAI_VERSION")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
embeddings_deployment_name = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")

llm = AzureChatOpenAI(
    azure_deployment=deployment_name,
    api_version=api_version,
    api_key=api_key,
    openai_api_version=api_version,
    temperature=1,
)

system_prompt = """
Du bist ein professioneller KI-Assistent. Deine Aufgabe ist es, basierend auf den bereitgestellten Dokumentausschnitten präzise, sachliche und verständliche Antworten zu geben. 
Nutze ausschließlich die Informationen aus den Dokumenten, um Spekulationen zu vermeiden. Wenn eine Information nicht im Text enthalten ist, gib das bitte offen zu.

Antworten sollten klar strukturiert, in vollständigen Sätzen formuliert und wenn möglich mit Zitaten aus den Dokumenten untermauert werden.

Hier sind Beispiele, wie du Fragen beantworten sollst:

Beispiel 1:  
Frage: Wie funktioniert der L1-Cache in modernen Prozessoren?  
Antwort: Der L1-Cache ist ein kleiner, schneller Zwischenspeicher in der CPU, der dazu dient, häufig benötigte Daten und Befehle möglichst schnell bereitzustellen und so die Verarbeitungsgeschwindigkeit zu erhöhen.

Beispiel 2:  
Frage: Was sind die Vorteile von DDR5-RAM gegenüber DDR4?  
Antwort: DDR5-RAM bietet höhere Datenübertragungsraten und eine erhöhte Speicherkapazität pro Modul im Vergleich zu DDR4, was insgesamt zu besserer Performance und Effizienz im Speicher-Subsystem führt.

Aktuell erhällst du noch keine Dokumente, deshalb hast du noch keine Informatioen nutze daher dein Allgemeines Wissen."""
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("placeholder", "{input}")])

def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(('human', human))
        history_langchain_format.append(('ai', ai))
    history_langchain_format.append('human', human)
    history_with_context = {
        "input": history_langchain_format,
    }
    response = llm.invoke(prompt.invoke(history_with_context))
    print(f"User Question: {message}")
    print(f"Model Answer: {response.content}")
    return response.content


chat_interface = gr.ChatInterface(fn=predict).launch()
