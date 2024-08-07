# document-ai-workshop
Das hier iat das Repository zu einen Workshop.
**GOAL** Wir werden eine [Retrieval Augmented Generation (RAG)](https://python.langchain.com/v0.2/docs/tutorials/rag/) App in Langchain bauen.  
Als zusätzliche Funktionen sind geplant
- Chat mit Dokumenten
- Chat Historie
- Gradio Chat Interface   


## Environment Setup

Install Python 3.10: 

Install [miniconda](https://docs.anaconda.com/miniconda/miniconda-install/) (Package Manager for Data Science Packages)

Setup Virtual Environment (venv) in terminal :  
```python -m venv documentai```

Setzt in Python Tab in VS Code die angelegte venv als Asufühurngsenvironnment.

## Einführung 
Kurze nach erzählung dr Workshop einfühunrg

### Was ist ein Retrieval Augmented Generation (RAG) App

Eine Ai Modell ist Input Output system. Wir gebn Frage ein und es generiert uns de wahrscheinlichste Anwtort auf dieser Basis. Im Grunde kennt man es aus der Mathematik: Du hast eine Function gibst eine Eingabe in die Paramter, die Funktion führt eine Operation aus und du erhälst das Ergebnis

TODO: Diagra wie funktioniert KI

Die Eingabe an die KI nennt man Prompt.

### Was erhält die KI als Eingabe?

**Einagbe**: Prompt + System Prompt

**Prompt**: Eingabe des Nutzer, der befejl oder die Aufgabe

**System Prompt**: Model Finetuning der RAG App Entwickler & Weitere Eingaben z.B. Dokumente  
Das Model finetuning kann z.B. Auflagen für das KI Modell sein, auf die Eingabe auf bestimmte Arten zu benatworten oder die Anfragen im Prompt in eine bestimmte Licht zu sehen. Genauso kann man hier versuchen dem Modell bestimmtes Verhalten zu verbieten. 

Und dann wären da noch die [LLM Konfigurations-Hyperparameter](https://learnprompting.org/de/docs/basics/configuration_hyperparameters), im Grunde genommen zahlenbasiete Stellschrauben, um die Ausgabe des Modells weiter zu beeinflussen.

### Get an OpenAI API Key
- First, navigate to https://platform.openai.com/account/api-keys
- Then, sign up for or sign into your OpenAI account.
- Click the Create new secret key button. It will pop up a modal that contains a string of text like this:
![GPT API Key Example](assets/gpt_api_key_image.png)
- Last, create in project-root a new file `my_gpt.key` and paste your key into it. (so you can use helper function: loadGptKey)

 Note that OpenAI charges you for each prompt you submit through these embeds. If you have recently created a new account, you should have 3 months of free credits. If you have run out of credits, don't worry, since using these models is very cheap. ChatGPT only costs about $0.02 for every seven thousand words you generate. from [learnprompting.org](https://learnprompting.org/de/docs/basics/embeds)


### Langchain: 


### Prompt Techniken: Wie interagiere ich effektiv mit der AI
Die Webseite [learnprompting.org](https://learnprompting.org/de/docs) erklärt wie Prompting funktioniert und stellt Ansätze dar wie man der KI aus der Perspektive der Prompt Eingabe bessere Ergebnisse entlocken kann. 










