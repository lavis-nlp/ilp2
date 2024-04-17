import json

class run_config():
    tail_templates:dict
    head_templates:dict
    system_prompt:str

    def __init__(self,prompt_templates_path:str,system_prompt_path):
        config_data = load_prompts(prompt_templates_path,system_prompt_path)
        self.tail_templates = config_data[0]
        self.head_templates = config_data[1]
        self.system_prompt = config_data[2]

    def get_tail_prompt(self,vertex:str,relation:str) -> str:
        prompt = self.system_prompt+" "+self.tail_templates[relation].format(vertex)
        return prompt
    
    def get_head_prompt(self,vertex:str,relation:str) -> str:
        prompt = self.system_prompt+" "+self.head_templates[relation].format(vertex)
        return prompt
        
def load_prompts(prompt_templates_path:str,system_prompt_path) -> (dict, dict, str):
    with open(prompt_templates_path) as file:
        json_file = json.load(file)
        tail_templates = json_file["tail"]
        head_templates = json_file["head"]
        assert type(tail_templates) == dict and type(head_templates) == dict

    with open(system_prompt_path) as file:    
        system_prompt = json.load(file)["system"]
        assert type(system_prompt) == str

    return tail_templates, head_templates, system_prompt