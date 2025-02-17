import os
import sys
import inspect
import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma

# Ignore: Fügt root Ordner für utils zum sys.path hinzu, damit es iportiert werden kann
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
rootdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(rootdir)
from utils import formatDocs, loadDocumentsFromDirectory  # noqa: E402

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
Bitte antworte mir immer auf deutsch. Bleibe immer höflich und professionell.

Bitte beantworte die Frage mit dem gegebenen Kontext zwischen "<context></context>". 
Wenn context keine relevanten Informationen zur Frage enthält, erfinde nichts und sage "Ich weiß die Antwort nicht.". 
<context>
{context}
</context>
"""
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
few_shot_structured_llm = prompt | llm


documents = loadDocumentsFromDirectory("SOURCE_DOCUMENTS")

# Embeddings ist our AI Modell. It will transfer our documents into an semantic Vector interpretation interpretation.
# A number based vector representation makes easier for the AI to understand semantic similiarity of text Passagen
# from https://python.langchain.com/docs/how_to/vectorstores/
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=embeddings_deployment_name,
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    openai_api_version=api_version)

# transform our Documents in an vectore store in a chromaDB while holding a document refence
db = Chroma.from_documents(documents, embeddings)

# query for identify the relevant Documents
query = "Was hat Kyros II. erobert"

# similiarity search from https://python.langchain.com/docs/how_to/vectorstores/#similarity-search
docs = db.similarity_search(query)

# We can also transform our Query to an vector interpretation
# similiarity searchbyVector from https://python.langchain.com/docs/how_to/vectorstores/#similarity-search
embedding_vector = embeddings.embed_query(query)
docs_embeded_query = db.similarity_search_by_vector(embedding_vector)

doc_content = formatDocs(docs)

# Gradio client predict functions, will be executed when User submit action in client
def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    history_with_context =  {
        "context": doc_content,
        "input": history_langchain_format,
    }

    print(history_with_context)
    response = few_shot_structured_llm.invoke(history_with_context)
    print(f"User Question: {message}")
    print(f"Model Answer: {response.content}" )
    return response.content


chat_interface = gr.ChatInterface(fn=predict).launch()
