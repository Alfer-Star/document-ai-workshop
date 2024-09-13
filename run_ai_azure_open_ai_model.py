import gradio as gr
from dotenv import load_dotenv
import os

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

system_prompt = """You are a good AI. You help solving my task or question!

Question: What is 1+1?
Answer: the Answer is 2.

Question: Which color has my blue cat?
Answer: The color of your cat is blue.

Question: Who won the Battle of Thermopylae in greece during the Persian Wars in 460 BC?
Answer: The battle was fought between the Persian empire and a coalition of greek city states. The Persian empire won.

Question: Who was the first King of Persia?
Answer: The first King and founder of Persia was Kyros II.
"""

prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
few_shot_structured_llm = prompt | structured_llm


def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    response = llm.invoke(history_langchain_format)
    print(f"User Question: {message}")
    print("Model Answer: " + response.content)
    return response.content


chat_interface = gr.ChatInterface(fn=predict).launch()
