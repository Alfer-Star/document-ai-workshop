from typing import List
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


def loadGptKey():
    f = open("./my_gpt.key", "r")
    return f.read()


def loadDocumentsFromDirectory(dirctory_path="./SOURCE_DOCUMENTS/"):
    """load directory content, do not use as AI Input, need some preperations
    install: pip install "langchain-unstructured[local]"
    from https://python.langchain.com/v0.2/docs/how_to/document_loader_directory/
    File Support (.md, .pdf, .csv ...): https://docs.unstructured.io/open-source/introduction/supported-file-types
    """
    loader = DirectoryLoader(dirctory_path)
    docs = loader.load()
    print("Successfull loaded Documents: " + str(len(docs)))
    return docs


def loadSingleMarkdownDocument(file_path: str):
    """loads a single Document, please do not use to big files as AI Input, they need to be splitted
    install: pip install "unstructured[md]" nltk
    from https://python.langchain.com/docs/how_to/document_loader_markdown/"""
    loader = UnstructuredMarkdownLoader(file_path)
    doc = loader.load()
    assert len(doc) == 1
    doc_content = doc[0].page_content
    print("Successfull loaded: " + file_path)
    print(doc_content[:250])
    return doc_content


def formatDocs(doc_list: List):
    """
    Formats Docs into Strings
    """
    doc_content = ""
    for doc in doc_list:
        title = doc.metadata["source"]
        pageContent = doc.page_content
        doc_content += f""" {title}
            {pageContent}
   
            """
    return doc_content

""" Load Wikipedia Pages According given Query 
    Does not work in Unternehmensnetzwerk, would need a proxy
"""
def loadWikipediaPages(searchQuery, top_k_results=3):
    w2 = WikipediaAPIWrapper(lang='de', top_k_results=top_k_results)
    return w2.load(searchQuery)


## Implement helper functions here, to keep the main code cleaner and simple to read
