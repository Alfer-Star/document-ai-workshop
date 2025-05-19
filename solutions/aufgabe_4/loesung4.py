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
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("placeholder", "{input}")])



documents = loadDocumentsFromDirectory("SOURCE_DOCUMENTS")

# Before we transform our Documents into an Vector Store we cut it into pieces for an simplier semantic understandig
# Text Splitter from https://python.langchain.com/v0.2/docs/how_to/recursive_text_splitter/
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
doc_pieces = text_splitter.split_documents(documents)
print("Successfull splitted Documents. Number of Chunks: " + str(len(doc_pieces)))

# Embeddings is our AI model. It will transfer our documents into a semantic Vector interpretation.
# A number based vector representation makes easier for the AI to understand semantic similiarity of text Passagen
# from https://python.langchain.com/docs/how_to/vectorstores/
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=embeddings_deployment_name,
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    openai_api_version=api_version
)

# transform our Documents in a vectore store in a chromaDB while holding a document refence
vectorStore = Chroma.from_documents(doc_pieces, embeddings)

# query to identify the relevant documents
query = "Bitte gebe mir nur Dokumente zurück, die etwas mit Harry Potter zu tun haben."

# similiarity search from https://python.langchain.com/docs/how_to/vectorstores/#similarity-search
docs = vectorStore.similarity_search(query)
print(f"Anzahl der Dokumente nach Similarity Search: {len(docs)}")

# We can also transform our Query to a vector interpretation
# similiarity searchbyVector from https://python.langchain.com/docs/how_to/vectorstores/#similarity-search
embedding_vector = embeddings.embed_query(query)
docs_embeded_query = vectorStore.similarity_search_by_vector(embedding_vector)

doc_content = formatDocs(docs)

def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    history_with_context = {
        "context": doc_content,
        "input": history_langchain_format,
    }
    response = llm.invoke(prompt.invoke(history_with_context))
    print(f"User Question: {message}")
    print(f"Model Answer: {response.content}")
    return response.content


chat_interface = gr.ChatInterface(fn=predict).launch()
