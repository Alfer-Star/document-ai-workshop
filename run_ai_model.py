from utils import *

import os

os.environ["OPENAI_API_KEY"] = loadGptKey()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

