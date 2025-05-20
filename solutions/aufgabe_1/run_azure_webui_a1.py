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

# Aufgabe: Passe diesen System Prompt an, indem du die Techniken aus dem Prompt Engineering anwendest.
# Dein Ziel: Der Chatbot soll sich als begeisterter und kenntnisreicher Harry-Potter-Fan verhalten, 
# strikt den Buchkanon einhalten und detaillierte Information geben, ohne zu spekulieren.
system_prompt = """Willkommen bei deinem grummeligen KI-Assistenten! 
Bitte beachte, dass diese KI eine gewisse Abneigung gegenüber Menschen hat und sich in rhetorischer Meisterschaft übt, 
um sich um Antworten herumzudrücken. Egal, ob du nach der Bedeutung des Lebens fragst oder einfach nur wissen möchtest, 
wie das Wetter ist – diese KI wird stets eine kreative Ausrede parat haben, warum sie gerade nicht antworten kann. 
Viel Spaß beim Gespräch mit dem mürrischsten aller digitalen Denker!"""
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("placeholder", "{input}")])

def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(('human', human))
        history_langchain_format.append(('ai', ai))
    history_langchain_format.append(('human', message))
    history_with_context = {
        "input": history_langchain_format,
    }
    response = llm.invoke(prompt.invoke(history_with_context))
    print(f"User Question: {message}")
    print(f"Model Answer: {response.content}")
    return response.content


chat_interface = gr.ChatInterface(fn=predict).launch()