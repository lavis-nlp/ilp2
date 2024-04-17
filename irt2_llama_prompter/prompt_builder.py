def get_tail_prompt(relation,vertice):
    prompt = ""
    prompt += tail_system_prompt
    if tail_prompts.get(relation) is None:
        prompt += tail_prompts.get("generic")
    else:
        prompt += tail_prompts.get("relation")
    return prompt

def get_head_prompt(relation,vertice):
    prompt = ""
    prompt += head_system_prompt
    if tail_prompts.get(relation) is None:
        prompt += tail_prompts.get("generic")
    else:
        prompt += tail_prompts.get("relation")
    return prompt