@function
def question(s, question_1):
    s += system("[INST] [SYS]")
    s += system("You are only able to answer naming the answers! \
                Dont write a single sentence! \
                Please give all answers! \
                Dont elaborate! \
                Never repeat the questions! \
                Never respond in sentences! \
                Dont bring new questions in context with old ones! \
                Separate the answers using a comma.")
                ##Answer in python string list format with the string elements being the answers!")
    s += system("[/SYS]")

    s += user(question_1)

    s += assistant(gen("answer_1", max_tokens=256))

    s += system("[/INST]")

set_default_backend(RuntimeEndpoint("http://localhost:30000"))

