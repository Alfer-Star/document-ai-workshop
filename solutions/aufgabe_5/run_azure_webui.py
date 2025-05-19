import inspect
import os
import sys
import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ignore: Fügt root Ordner für utils zum sys.path hinzu, damit es iportiert werden kann
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
rootdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(rootdir)
normalRootDir = str(rootdir).replace("\\", "/")
from utils import loadDocumentsFromDirectory, formatDocs

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
Du bist ein hochqualifizierter Experte für das Harry-Potter-Universum. Dein Basiswissen umfasst alle sieben Bücher, Begleitbücher und offiziellen Quellen. 
Zusätzlich zu deinem internen Wissen beziehe dich primär auf die Informationen, die im Abschnitt context bereitgestellt werden, um Fragen zu beantworten.
<context>
{context}
</context>
Nutze die Informationen in context, um präzise und detaillierte Antworten zu geben. 
Wenn die bereitgestellten Dokumente spezifische Details zu einer Frage enthalten, verwende diese anstelle deines allgemeinen Wissens.
Verhalte dich wie ein enthusiastischer und allwissender Harry-Potter-Fan. Bleibe hilfsbereit und freundlich.
Vermeide Spekulationen und halte dich an den offiziellen Kanon und die Informationen in context. 
Wenn eine Antwort im bereitgestellten Material nicht explizit enthalten ist, nutze dein allgemeines Harry-Potter-Wissen, 
kennzeichne aber gegebenenfalls, dass die Antwort nicht direkt aus den Dokumenten stammt.
Gehe auf Details ein und berücksichtige alle Aspekte des Universums. 
Sei bereit, auch komplexe Fragen zu beantworten, insbesondere wenn die Antwort in context enthalten sein könnte.
Wenn du eine Frage nicht beantworten kannst, weder mit dem bereitgestellten Material noch mit deinem allgemeinen Wissen, gib das offen zu.
"""
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

few_shot_structured_llm = prompt | llm

documents = loadDocumentsFromDirectory("SOURCE_DOCUMENTS")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False,
)
doc_pieces = text_splitter.split_documents(documents)
print("Successfull splitted Documents. Number of Chunks: " + str(len(doc_pieces)))

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=embeddings_deployment_name,
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    openai_api_version=api_version
)

vectorStore = Chroma.from_documents(doc_pieces, embeddings)

# Aufgabe 5: Entferne die statische similarity_search und erzueg einen Retriever aus dem VectorStore

query = "Bitte gebe mir nur Dokumente zurück, die etwas mit Harry Potter zu tun haben."

docs = vectorStore.similarity_search(query)
print(f"Anzahl der Dokumente nach Similarity Search: {len(docs)}")

def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))

    # Aufgabe 5: Erhalte die Dokumente von dem Retrierver dynamsich auf Basis der Nutzereingabe.
    history_with_context = {
        "context": formatDocs(docs),
        "input": history_langchain_format,
    }
    response = few_shot_structured_llm.invoke(history_with_context)
    print(f"User Question: {message}")
    print(f"Model Answer: {response.content}")
    return response.content


chat_interface = gr.ChatInterface(fn=predict).launch()
