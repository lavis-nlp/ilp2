import json
import csv
import datetime
from pathlib import Path

import irt2
from irt2.dataset import IRT2
from irt2.dataset import MID

class run_config():
    irt2_data_path:str

    tail_templates:dict
    head_templates:dict
    system_prompt:str
    data:irt2.dataset.IRT2

    def __init__(self,tail_templates:dict,head_templates:dict,system_prompt:str,irt2_data_path:str):
        self.tail_templates = tail_templates
        self.head_templates = head_templates
        self.system_prompt = system_prompt
        self.irt2_data_path = irt2_data_path
        self.data = IRT2.from_dir(path=irt2_data_path)

    # Gibt fertigen tail-completion-prompt zurück
    def get_tail_prompt(self,vertex:str,relation:str) -> str:
        prompt = self.system_prompt+' '+self.tail_templates[relation].format(vertex)
        return prompt
    
    # Gibt fertigen head-completion-prompt zurück
    def get_head_prompt(self,vertex:str,relation:str) -> str:
        prompt = self.system_prompt+' '+self.head_templates[relation].format(vertex)
        return prompt
    
    # Speichert run_config mit Namen config_name im Ordner path, default: run_configurations
    def export(self,config_name:str,path:Path=Path('run_configurations')):
        if not path.exists():
            path.mkdir()
        json_path = path / config_name
        data = {
            'tail_prompt_templates':self.tail_templates,
            'head_prompt_templates':self.head_templates,
            'system_prompt':self.system_prompt,
            'irt2_data_path':self.irt2_data_path
        }
        with open(json_path,'w') as json_file:
            json.dump(data,json_file,indent=4)

        print('Run-Configuration nach \'{}/{}\' exportiert'.format(path,config_name))

    # Gibt Infos zur run_config
    # TODO
    def info(self):
        s = "-"*20+"CONFIG"+"-"*20+"\n"
        s += 'System-Prompt: {}\n\n Tail Prompt Templates: {}\n\n Head Prompt Templates: {}\n\n IRT2-Data: {}\n'.format(self.system_prompt,self.tail_templates,self.head_templates,self.irt2_data_path)
        s += "-"*46
        print(s)

    # TODO
    # Test laufen lassen
    def run_test(self):
        dir:Path = create_result_folder()
        self.export('run_config',dir)

# Ordner für Testergebnisse anlegen, Pfad zurückgeben
def create_result_folder() -> Path:
    path = Path('results')
    if not path.exists():
        path.mkdir()
    formatted_date_time = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    folder_name = 'result_{}'.format(formatted_date_time)
    dir_path = path / folder_name
    i = 0
    while dir_path.exists():
        i += 1
        formatted_date_time = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        folder_name = 'result_{}_({})'.format(formatted_date_time,i)
        dir_path = path / folder_name
    dir_path.mkdir()
    return dir_path

# Config erstellen aus Daten
def create_config(prompt_templates_path:str,system_prompt_path:str,irt2_data_path:str) -> run_config:
    templates = load_prompt_templates(prompt_templates_path)
    tail_templates = templates[0]
    head_templates = templates[1]
    system_prompt = load_system_prompt(system_prompt_path)
    return run_config(tail_templates,head_templates,system_prompt,irt2_data_path)

# Importert run_config mit namen config_name as Ordner path
def import_config(config_name:str,path:Path=Path('run_configurations')) -> run_config:
    if not path.exists:
        print("Kein Ordner, aus dem Importiert werden könnte!")
        return
    json_path = path / config_name
    with open(json_path) as file:
        json_file = json.load(file)
        tail_templates = json_file['tail_prompt_templates']
        head_templates = json_file['head_prompt_templates']
        system_prompt = json_file['system_prompt']
        irt2_data_path = json_file['irt2_data_path']
        config = run_config(tail_templates,head_templates,system_prompt,irt2_data_path)
    return config

# Läd Systemprompt von Pfad
def load_system_prompt(system_prompt_path) -> (str):
    with open(system_prompt_path) as file:    
        system_prompt = json.load(file)['system']
        assert type(system_prompt) == str

    return system_prompt

# Läd Prompt-Templates von Pfaden
def load_prompt_templates(prompt_templates_path:str) -> (dict, dict):
    with open(prompt_templates_path) as file:
        json_file = json.load(file)
        tail_templates = json_file['tail']
        head_templates = json_file['head']
        assert type(tail_templates) == dict and type(head_templates) == dict

    return tail_templates, head_templates