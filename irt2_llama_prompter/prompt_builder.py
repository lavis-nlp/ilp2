import json

#class prompt_builder():
tail_prompts = dict
head_prompts = dict

tail_system_prompt = str
head_system_prompt = str

def load_prompts(prompt_file):
    json_file = json.load(prompt_file)
    self.tail_prompts = json_file["tail"]
    self.head_prompts = json_file["head"]

def load_system_prompt(sys_prompt_file):
    json_file = json.load(sys_prompt_file)
    self.tail_system_prompt = json_file["tail"]
    self.head_system_prompt = json_file["head"]

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

