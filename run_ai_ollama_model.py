from langchain.schema import AIMessage, HumanMessage
import gradio as gr

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

system_prompt = "You are a helpful assistant that translates {input_language} to {output_language}."

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        ("human", "{input}"),
    ]
)

model = OllamaLLM(model="llama3.1", temperature=0,)

llm = prompt | model

## end TODO: make it work with gradio

def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    response = llm(history_langchain_format)
    print("User Question: {message}")
    print("Model Answer:")
    print(response)
    return response.content

gr.ChatInterface(predict).launch()
