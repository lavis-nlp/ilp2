from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint

def __init__(self):
    self.temp = 0
    self.tokens = 256

def prompt_model(s,system_prompt,question):
    s += system("[INST] [SYS]")
    s += system(system_prompt)
    s += system("[/SYS]")

    s += question

    s += assistant(gen("answer"),max_tokens=tokens,temperature=temp)
    s += system("[/INST]")

def set_temp(t):
    temp = t

def set_tokens(t):
    tokens = t      

def connect(port):
    set_default_backend(RuntimeEndpoint("http://localhost:"+str(port)))  
