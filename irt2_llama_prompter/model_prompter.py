from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint

temp = 0
tokens = 256
ready = False

def prompt_model(s_prompt,q_prompt):
    state = question.run(
        system_prompt=s_prompt,
        question=q_prompt
    )
    answers = state["answer"]
    return answers

def parse_answer(answers):
    result = answers.split(",")
    return result

@function
def question(s,system_prompt,question):
    s += system("[INST] [SYS]")
    s += system(system_prompt)
    s += system("[/SYS]")

    s += question

    s += assistant(gen("answer",max_tokens=tokens,temperature=temp))
    s += system("[/INST]")

def set_temp(t):
    temp = t

def set_tokens(t):
    tokens = t      

def connect(port):
    set_default_backend(RuntimeEndpoint("http://localhost:"+str(port)))
    ready = True
