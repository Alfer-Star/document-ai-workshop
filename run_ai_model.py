from utils import *

import os

from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import openai
import gradio as gr

# init ChatGPT API LLM Verbindung

os.environ["OPENAI_API_KEY"] = loadGptKey() # Replace with your key

llm = ChatOpenAI(temperature=1.0, model="gpt-4o-mini")

def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = llm(history_langchain_format)
    return gpt_response.content

gr.ChatInterface(predict).launch()
