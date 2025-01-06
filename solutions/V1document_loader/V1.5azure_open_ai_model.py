import gradio as gr
from dotenv import load_dotenv
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import loadDocumentsFromDirectory, loadSingleMarkdownDocument

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
Bitte antworte mir immer auf deutsch. Bleibe immer höfflich und professionell.

Bitte beantworte die Frage mit dem gegebenen context. 
Wenn context keione relevanten Informationen zur Frage enthält, erfinde nichts und sage "Ich weiß die Antwort nicht. :(": 
<context>
{context}
</context>
"""
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
few_shot_structured_llm = prompt | structured_llm


## doc_content = loadSingleMarkdownDocument("SOURCE_DOCUMENT/kyros_ii_persia_history.md")
documents = loadDocumentsFromDirectory("SOURCE_DOCUMENTS")
#Text Splitter from https://python.langchain.com/v0.2/docs/how_to/recursive_text_splitter/
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.split_documents(documents)
print("Successfull splitted Documents. Number of Chunks: " + len(texts))


def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    historyWithContext =  {
        "context": texts.pop(),
        "input": history_langchain_format,
    }
    print(historyWithContext)
    response = few_shot_structured_llm.invoke(historyWithContext)
    print("User Question: {message}")
    print("Model Answer: " + response.content)
    return response.content


chat_interface = gr.ChatInterface(fn=predict).launch()
