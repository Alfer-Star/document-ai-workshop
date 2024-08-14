# document-ai-workshop (Work in progress)
Das hier ist das Repository zu einen Workshop.  

**GOAL** Wir werden eine [Retrieval Augmented Generation (RAG)](https://python.langchain.com/v0.2/docs/tutorials/rag/) App in Langchain bauen. In einfachen Worten einen ChatBot, dem wir zusätzliches Wissen zu Verfügung stellen
Als Funktionen sind geplant
- Chat mit Dokumenten (ChatBot der Wissen aus Dokumenten ziehen kann)
- (Chat Historie) Fertig
- (Graphical Chat Interface) Fertig

Das Repository stellt eine simple Ki Applikation mit einer GUI zur Verfügung. Es eine Anleitung zum Setup und eine kurze Einführung in das Thema mit Links. 

TODO: english version  
## Environment Setup

Install Python 3.10: [python.org](https://www.python.org/downloads/release/python-31011/) or via [Microsoft Store](https://apps.microsoft.com/detail/9pjpw5ldxlz5?hl=en-US&gl=US)

Folgendes Schritte musst du im Terminal ausführen, dabei solltest du dich im Projekt Directory befinden
![correct_directory](assets/terminal_project_directory.PNG)

Setup **Virtual Environment (venv)** in terminal:  
```python -m venv documentai```  
Hier werden wir die Pakete in ein abgeschirmten Umgebung installieren, dem "venv".

Nun müssen wir das venv *documentai* im terminal aktivieren.
```activate documentai```  
*Wichtig!* Die KI Applikation wird sich nur in der Konsole ausführen lassen, wenn das Venv aktiv ist!
![Aktives venv](assets/active_venv.png)  
(*Alternative*) venv via [VSCode Plugin Python EnvironManager](https://marketplace.visualstudio.com/items?itemName=donjayamanne.python-environment-manager) erstellen & aktivieren  
(*Alternative*) venv via [VSCODE Commands (+ Erklärungen)](https://code.visualstudio.com/docs/python/environments) aktivieren

Installiere benötigte Abhängigkeiten & Pakete mit dem Befehl:
`pip install -r requirements.txt`

Die KI Applikation ist jetzt fast bereit zum ausführen. Wir haben da zwei Varianten.
- OpenAi Playground basierte Ki Applikation (*Cloud* KI Modell) `python run_ai_gpt_model.py`
- Ollama basierte Ki Applikation (*lokales* KI Modell) => `python run_ai_ollama_model.py`

### OpenAi Ki Applikation (Cloud)
Das KI Modell (ChatGPT) läuft in der Cloud von Open AI. wir sprechen dessen API an.
Damit das funktioniert müssen wir uns einen Key generieren lassen.
#### Get an OpenAI API Key

- First, navigate to https://platform.openai.com/account/api-keys
- Then, sign up for or sign into your OpenAI account.
- Click the Create new secret key button. It will pop up a modal that contains a string of text like this:
![GPT API Key Example](assets/gpt_api_key_image.png)
- Last, create in project-root a new file `my_gpt.key` and paste your key into it. (so you can use helper function: loadGptKey)

Note that OpenAI charges you for each prompt you submit through these embeds. If you have recently created a new account, you should have 3 months of free credits. If you have run out of credits, don't worry, since using these models is very cheap. ChatGPT only costs about $0.02 for every seven thousand words you generate.  
From [learnprompting.org](https://learnprompting.org/de/docs/basics/embeds)

**Applikation Starten**: 
`python run_ai_gpt_model.py`

### Ollama Ki Applikation 
Häufig ist es so, dass man den großen KI Anbietern wie OpenAI nicht vertraut, weil bei KI Interaktionen Daten entstehen, welche die zum trainieren ihres Modells wider verwenden. KI-Modelle kann man auch lokal auf den eigenen Rechner auszuführen und mit ihnen zu interagieren. 

#### Anforderungen an deinen Computer 
 
KI Modelle sind sehr Computer Ressourcen hungrig.   
Grundsätzlich gilt: Die meisten Modelle lassen sich ausführen, wenn dein Computer eine NVIDIA Grafikarte verbaut hat.
Sonst gilt für Modelle kleine Modelle der Größe 7 Mrd. Parameter (7B) 
- Guter Prozessor (Intel i7), sonst generiert die KI möglicherweise ein Wort in 30s 
- Großer Arbeitsspeicher (RAM) (Minimum 8GB, besser 16 und 32 GB); KI-modelle werden in den RAM geladen, ein 8GB großes Modell lässt sich nicht 8GG RAM laden, kein Platz; selbst 4Gb sind dann schon kritisch.
Ressourcen hunger steigt mit der Größe des Modells. Es gilt Llama3 7B => Kleines Modell, Llama3 70B => großes Modell. 70b steht für 70 Mrd Paramedter. Viele Modelle haben eine große und eine kleine Ausführung, wie Llama3 von Meta.

#### Ollama 
Prominentes Tool um Modelle von der Plattform [HuggingFace](https://huggingface.co) (Die Quelle7Community für KI Modelle, idR. Open Source) auszuführen ist [llama.cpp](https://github.com/ggerganov/llama.cpp). llama.cpp ist aber nicht Anfänger freundlich, daher verwenden wir [Ollama](https://ollama.com). Die gleiche Technologie, jedoch vereinfachte Installation und Konfiguration. Nachteil wir können nur Moddell aus dieser [Liste](https://ollama.com/library) installieren.


1. Install Ollama: https://ollama.com/download 
2. [Browse Models](https://ollama.com/library) for your use Case or use a recommendation model
3. Load Model herunter via `ollama pull <model>`
4. Start Modell via `ollama run <model>` (optional)
5. Run ``python run_ai_ollama_model.py``

#### Modell Empfehlungen

- [Mistral](https://ollama.com/library/mistral)
- [LLama3 by Meta](https://ollama.com/library/llama3.1)
- [gemma2 by Google](https://ollama.com/library/gemma2)

### Applikation startet eine simple GUI
Applikation im Terminal starten:  
![alt text](assets/run_app_interminal.png)  
Gradio Web GUI öffnen: http://localhost:7860
![alt text](assets/gradio_test_kyros.png)

## Einführung KI

### Was sich Ki-modelle und wie entsteht diese künstliche "Intelligenz"

Eine KI Modell ist Input Output System. *Wir geben eine Anweisung und es generiert uns ein Ergebnis.*   
Im Grunde kennt man es aus der Mathematik: Wie eine mathematische Funktion nimmt das KI Modell Eingabe Parameter entgegen, die Eingabe Parameter werden von der Funktionsoperation (die Rechnung) in das Ergebnis umgewandelt. z.B. ``f(x)=>x+x f(2)=4``.  
Das ist das grundlegende Prinzip, wie die KI Funktioniert. 

Grundsätzlich Vorgang wie ein KI Modell zu einem Ergebnis kommt generiert,schwer nachzuvollziehen. Ähnlich wie wenn du versuchst in das Gehirn deines Nachbarn zu schauen, um dir seine letzte Handlung zu erklären. Das Gehirn ist ein ebenso komplexes Modell, das auf die eingegeben Reize (sinne: Augen, Ohren usw.) getriggert wird and antrainiert reagiert.

**Um KI Modelle zu erstellen haben wir das menschl. Gehirn vereinfacht nachgebaut.** Die Wundertechnik schimpft sich Neuronale Netze und gibt es schon seid über 30 Jahren eingesetzt. 
KI Modelle müssen trainiert werden, bevor sie Ergebnisse generieren könne. Wie ein Mensch lernen bzw. Erfahrungen sammeln muss. Eine KI wird mit einen großen Datensatz trainiert, die zu ihren Anwendungsfall passen. Das Neuronale Netz erkennt bzw. lernt iterativ wiederkehrende Strukturen bzw. Muster. Einfach gesagt erfüllt die Ki Anweisungen aus einen separaten Testdatensatz und konstruiert anfangs chaotisch etwas aus den Trainingsdaten zu einen Ergebnis zusammen. Das Modell lernt indem es für gute Ergebnisse, für schlechte bestraft wird. Auf diese Weise lernt das das Modell, welch Kombinationen gut sind und welch vermieden werden soll. (sehr vereinfacht, Lernprozess ist beeinflussbar => Optimierung)  
Erreicht man eine Punkt an dem das KI-Modell korrekte Voraussagen erzeugt, ist die Ki fertig trainiert. Es wird vorgetäuschte Intelligenz geschaffen. Das Modell hat gelernt ein Muster im Datensatz zu erkenne und kann deshalb Voraussagen treffen, welche Rückgabe auf die Eingabe erwartete wird.

**Ein Modell ist stark von den Trainingsdaten abhängig und dessen erkennbaren wiederkehrenden Strukturen**. Denn diese reproduziert oder verknüpft sie zu etwas ableitbaren. Was die KI nicht kennt kann sie nicht generieren.  
**KI Modelle können auf spezielle Anwendungsfälle optimiert sein**. ChatGPT ist ein "allgemeines" Multimodales Modell. OpenAi ist bestrebt, das man es für jeden Anwendungsfall einsetzen kann.   
**Bias ist ein großes Problem**. Trainiere ein Modell mit einen sexistischen Datensatz und du erhältst eine sexistische KI. Vielleicht hast du das ja gewollt um es auf Sexismus zu spezialisieren?   
 **Datensatz bestehen häufig überwiegend aus englischsprachigen Daten**. Folglich wird das Modell besser English als Deutsch "verstehen". Auf unserer Welt wird mehr Daten in englisch als in deutsch erstellt. Demzufolge gibt auch mehr englisch basierte Daten zum trainieren. Bei den meisten großen Modellen (Meta, Google, OpenAI) ist das aber kein Problem mehr. Diese bezeichnet man als oft auch als **multilinguale Modelle**.

Wir werden in diesen Workshop Sprachmodelle (text2text) verwenden. KI-modelle, die auf Ein- und Ausgabe von Sprache trainiert worden sind. Man bezeichnet diesen großen Sprachmodellen als **Large Language Models (LLM)**.  Daneben gibt es eine Reihe anderer Arten Ki-Modelle , z.B.welche die Bilder erstellen.

Die Eingabe an die KI nennt man übrigens Prompt.

### Begriffe: Was erhält die KI als Eingabe? (Prompt)

**Modell Input** (Prompt): Nutzer Input (Prompt) + System Prompt + Prompt Template

**Prompt**: Eingabe des Nutzer, der Befehl oder die Aufgabe für KI, wird häufig als Prompt bezeichnet. Aber technisch gesehen besteht ein Prompt aus allen Eingaben an das KI-Modell. 

**System Prompt**: Modell Finetuning, im Grunde weitere Eingaben textuelle Eingaben. Das können zusätzliche Anweisungen sein, wie das KI-Modell die Nutzer Eingabe verarbeiten soll. z.B. kann man hier eingeben, dass das KI Modell seine Testausgabe wie Yoda aus Stars Wars formulieren soll. Ohne das der Nutzer explizit die Anweisung gibt, wie Yoda das Modell antworten wird.  
hier kann man auch dem Modell Beispeile geben wi

Und dann wären da noch die [LLM Konfigurations-Hyperparameter](https://learnprompting.org/de/docs/basics/configuration_hyperparameters). im Grunde genommen zahlenbasierte Stellschrauben, um die Ausgabe des Modells weiter zu beeinflussen. Diese gehören nicht zum Prompt, sind aber auch Eingaben an das Modell

### Retrieval Augmented Generation (RAG) Application
Was wir bauen werden nennt man in der Dachsprachen RAG Applikation. Eine Obendrauf Erweiterung eines fertig trainierten KI-Modells, zur Optimierung für einen bestimmt.
Man kann auch sagen bei einem **Retrieval Augmented Generation (RAG)** erweitern wir die bereits massive Wissensbasis des KI Modells, indem wir ihm zusätzlichen Informationen mitgeben. Z.B. indem wir die Eingabe durch die den Dokumenten unserer Firma oder den Suchergebnisse einer Google Search ergänzen.

### Embeddings: Dokumente für das AI Modell aufbereiten
Ein KI Modell kann nur eine Gewisse Menge an Informationen als Anweisung entgegebn nehmen. 
TODO: beschriebe Was sind Embeddings + Image

## Tools & Technologien
### Langchain: 
TODO: Ausführlicher
[How Tos](https://python.langchain.com/v0.2/docs/how_to/#tools)

#### Dokumenten einlesen & Embeddings in Langchain

[Retrieval Chain](https://python.langchain.com/v0.1/docs/modules/data_connection/)

#### Langchain Tools
AI Applikation mit zusätztlichen Fähigkeiten ausrüsten und die Kompetenzen von AI Modellen erweitern, meisten um mit der Welt drumherum zu interagieren.
[Tools Dokumentation](https://python.langchain.com/v0.1/docs/modules/tools/)

[Langchian Tool Übersicht](https://python.langchain.com/v0.2/docs/integrations/tools/)

TODO: Check 0.2 Api https://python.langchain.com/v0.1/docs/modules/data_connection/

Eigene Tools bauen ist auch möglich.

#### Langchain mit Ollama Links
[Anleitung Chat Einrichtung](https://python.langchain.com/v0.2/docs/integrations/chat/ollama/)
[Multi Modal Support](https://python.langchain.com/v0.2/docs/integrations/llms/ollama/#multi-modal)
[Using Model Tools (if exist)](https://python.langchain.com/v0.2/docs/integrations/chat/ollama/#tool-calling)
[Tooling with few Shoot Ansatz](https://python.langchain.com/v0.2/docs/how_to/tools_few_shot/)

### Gradio: Simple Web GUI for via Python Commands for Ai interaction
[Create a Chatbot with Gradio](https://www.gradio.app/guides/creating-a-chatbot-fast#using-your-chatbot-via-an-api)

[Ergänze Multimodale features zum Interface](https://www.gradio.app/guides/creating-a-chatbot-fast#add-multimodal-capability-to-your-chatbot) zum Beispiel File Upload


## Prompt Engeneering: Wie interagiere ich effektiv mit der AI
Die Webseite [learnprompting.org](https://learnprompting.org/de/docs) erklärt wie Prompting funktioniert und stellt Ansätze dar wie man der KI aus der Perspektive der Prompt Eingabe bessere Ergebnisse entlocken kann. 

### Few Shot Prompting
[Was ist few Shot Prompting?](https://learnprompting.org/de/docs/basics/few_shot)
In der Praxis dieses Workshops werden wir Few Shot Prompting verwenden, um in dem System Prompt dem KI Modell einen grobe Situation vorgegen. Im Grunde nur darum geht, dem Modell einige Beispiele für das zu zeigen bzw. Anweisungen zeigen (sogenannte Shots), was es tun soll.  
[Few Shot Prompting in Langchain](https://python.langchain.com/v0.2/docs/how_to/structured_output/#few-shot-prompting)

### Rollen Prompting






