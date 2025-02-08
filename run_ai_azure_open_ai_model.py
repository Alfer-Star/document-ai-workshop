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
"""
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
few_shot_structured_llm = prompt | structured_llm

""" 
    Gradio Client Interaction with AI-model Chain
    Will be executed everytime user submits his question in gradio Client 
    Includes the current history of the conversation
"""
def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    historyWithContext =  {
        "input": history_langchain_format,
    }

    # sends Ai user message and Ai response will be generated
    response = few_shot_structured_llm.invoke(historyWithContext)
    
    #print("User Question: {historyWithContext}")
    print(f"User Question: {message}")
    print(f"Model Answer: {response.content}" )
    
    return response.content


chat_interface = gr.ChatInterface(fn=predict).launch()
