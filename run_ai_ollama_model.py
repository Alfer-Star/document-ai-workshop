from langchain.schema import AIMessage, HumanMessage
import gradio as gr

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

system_prompt = "You are a helpful assistant."

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        ("human", "{input}"),
    ]
)

model = OllamaLLM(model="llama3.1")

llm = prompt | model

def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    response = llm.invoke(
        {
            "input": history_langchain_format,
        })
    print("User Question:")
    print(message)
    print("Model Answer:")
    print(response)
    return response

gr.ChatInterface(predict).launch()
