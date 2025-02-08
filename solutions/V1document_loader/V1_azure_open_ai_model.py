import os
import sys
import inspect

import gradio as gr
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

# Ignore: Fügt root Ordner für utils zum sys.path hinzu, damit es iportiert werden kann
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
rootdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(rootdir)
normalRootDir = str(rootdir).replace("\\", "/")
from utils import loadSingleMarkdownDocument  # noqa: E402


load_dotenv()

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_KEY")
api_version = os.getenv("AZURE_OPENAI_VERSION")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")

llm = AzureChatOpenAI(
    azure_deployment=deployment_name,
    api_version=api_version,
    api_key=api_key,
    openai_api_version=api_version,
    temperature=1,
)

structured_llm = llm.with_structured_output(AIMessage)

system_prompt = """
Bitte antworte mir immer auf deutsch. Bleibe immer höflich und professionell.

Bitte beantworte die Frage mit dem gegebenen Kontext zwischen "<context></context>". 
Wenn context keine relevanten Informationen zur Frage enthält, erfinde nichts und sage "Ich weiß die Antwort nicht.". 
<context>
{context}
</context>
"""
prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)
few_shot_structured_llm = prompt | structured_llm

doc_content = loadSingleMarkdownDocument("SOURCE_DOCUMENTS/kyros_ii_persia_history.md")


def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    historyWithContext = {
        "context": doc_content,
        "input": history_langchain_format,
    }
    print(historyWithContext)
    response = few_shot_structured_llm.invoke(historyWithContext)
    print(f"User Question: {message}")
    print(f"Model Answer: {response.content}" )
    return response.content


chat_interface = gr.ChatInterface(fn=predict).launch()
