from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint

temp = 0
tokens = 256
ready = False

def prompt_model(prompt:str) -> str:
    state = question.run(
        prompt
    )
    answers = state["answer"]
    return answers

def parse_answer(answers):
    result = answers.split(",")
    return result

@function
def question(s,prompt:str):
    s += prompt
    s += assistant(gen("answer",max_tokens=tokens,temperature=temp))
    
    
def set_temp(t):
    temp = t

def set_tokens(t):
    tokens = t      

def connect(port:int):
    set_default_backend(RuntimeEndpoint("http://localhost:"+str(port)))
    ready = True
