from utils import loadGptKey

import os

from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import gradio as gr

from langchain_core.prompts import ChatPromptTemplate

# init ChatGPT API LLM Verbindung

os.environ["OPENAI_API_KEY"] = loadGptKey() # Replace with your key

llm = ChatOpenAI(temperature=1.0, model="gpt-4o-mini")


# AIMessage class represents the structure of the answer we await from our Model. 
# See https://python.langchain.com/v0.2/docs/how_to/structured_output/#few-shot-prompting
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

prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("placeholder", "{input}")])

few_shot_structured_llm = prompt | structured_llm


def predict(message, history):
    """this function represents the interaction between the model an the Gradio Web GUI.
    this function called everytime when a user writes a message in his gradio chat confirms."""

    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = few_shot_structured_llm.invoke(history_langchain_format)
    #gpt_response = llm(history_langchain_format) # ask model directly
    print("User Question: {message}")
    print("Model Answer:")
    print(gpt_response)
    return gpt_response.content

gr.ChatInterface(predict).launch()
