import os
import gradio as gr
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=embeddings_deployment_name,
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    openai_api_version=api_version)

system_prompt = """
Bitte antworte mir immer auf deutsch. Bleibe immer höflich und professionell.

Bitte beantworte die Frage mit dem gegebenen Kontext zwischen "<context></context>".
Wenn context keine relevanten Informationen zur Frage enthält, erfinde nichts und sage "Ich weiß die Antwort nicht.".
<context>
{context}
</context>
"""
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("placeholder", "{input}")])



raw_documents = loadDocumentsFromDirectory("SOURCE_DOCUMENTS")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len, )
documents = text_splitter.split_documents(raw_documents)
print("chunks")
print(len(documents))
print(documents)
vector_store = Chroma.from_documents(documents, embeddings)

def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))

    # Use similarity_search_with_score to get documents and scores
    docs_with_scores = vector_store.similarity_search_with_score(message)

    print("Retrieved documents with scores:")
    doc_content_with_scores = ""
    relevant_docs = [] # To store docs above your score threshold if you want to apply one later
    source_documents_names = []

    # Sort documents by score in descending order
    sorted_docs_with_scores = sorted(docs_with_scores, key=lambda x: x[1])

    # Print only the top 2 scores (or fewer if less than 2 documents)
    top_scores_to_print = sorted_docs_with_scores[:min(7, len(sorted_docs_with_scores))]
    for doc, score in top_scores_to_print:
        print(f"Score: {1-score}")
        print(f"Document content: {doc.page_content}") # Print document content
        print("\n---")

    doc_content_with_scores = "" # Reset to ensure we are using ALL documents for context, not just top 2 for printing
    for doc, score in docs_with_scores: # Iterate through ALL documents for further processing
        doc_content_with_scores += f"Score: {score}\nDocument content: {doc.page_content}\n---\n" # For debugging and showing all doc content

        if score >= 0.0: # You can adjust this threshold as needed, 0.0 will include all docs.
            relevant_docs.append(doc) # Store docs above threshold if needed
            source_name = doc.metadata.get('source', 'Unbekanntes Dokument') # Default name if 'source' is not found
            source_documents_names.append(os.path.basename(source_name))

    # Format the relevant docs (or all docs if you want to pass all to context for debugging) for context
    unique_source_documents_names = sorted(list(set(source_documents_names))) # Sortierung ist optional, macht es aber oft übersichtlicher
    formatted_source_docs = ", ".join(unique_source_documents_names)
    formatted_docs = formatDocs(relevant_docs)
    history_with_context = {
        "input": history_langchain_format,
        "context": formatted_docs
    }
    response = llm.invoke(prompt.invoke(history_with_context))
    answer = response.content
    output_string = f"Ich habe in folgenden Dokumenten etwas gefunden: {formatted_source_docs}.\nAntwort: {answer}"
    print(f"User Question: {message}")
    print(f"Model Answer: {answer}")
    return output_string


chat_interface = gr.ChatInterface(fn=predict).launch()