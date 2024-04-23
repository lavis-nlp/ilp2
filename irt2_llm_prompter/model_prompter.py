from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint

# Parameter

# Macht Modell deterministisch
temp = 0

# Maximale Antwortlänge?
tokens = 256

# Checkt, ob connect ausgeführt wurde
ready = False

# Zum Prompten nutzen
# Returnt 
def prompt_model(prompt:str) -> str:
    state = question.run(
        prompt
    )
    answers = state["answer"]
    return answers

# Parsed die Antwort
#TODO
def parse_answer(answers):
    result = answers.split(",")
    return result

# Schickt Prompt an Modell und generiert Antwort
@function
def question(s,prompt:str):
    s += prompt
    s += assistant(gen("answer",max_tokens=tokens,temperature=temp))
    
    
def set_temp(t):
    temp = t

def set_tokens(t):
    tokens = t      

# Verbindet Prompter mit Sgl-Server
def connect(port:int):
    set_default_backend(RuntimeEndpoint("http://localhost:"+str(port)))
    ready = True
