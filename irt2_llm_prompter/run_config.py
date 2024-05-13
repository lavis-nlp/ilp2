import json
from pathlib import Path
from dataclasses import dataclass

from irt2_llm_prompter.customErrors import (
    MissingGenericException,
    MissingTemplatesException,
)


@dataclass
class RunConfig:
    data_type: str

    tail_templates: dict
    head_templates: dict
    system_prompt: str

    model_path: str
    tensor_parallel_size: int

    # data:irt2.dataset.IRT2

    def __init__(
        self,
        tail_templates: dict,
        head_templates: dict,
        system_prompt: str,
        data_type: str,
        model_path: str,
        tensor_parallel_size: int,
    ):
        self.tail_templates = tail_templates
        self.head_templates = head_templates
        self.system_prompt = system_prompt
        self.data_type = data_type
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size

    # Gibt fertigen tail-completion-prompt zurück
    def get_tail_prompt(self, mention: str, relation: str) -> str:
        if relation not in self.tail_templates:
            print("Used generic")
            content = self.tail_templates["generic"].format(mention, relation)
        else:
            content = self.tail_templates[relation].format(mention)

        prompt = "{} {}".format(self.system_prompt, content)
        return prompt

    # Gibt fertigen head-completion-prompt zurück
    def get_head_prompt(self, mention: str, relation: str) -> str:
        if relation not in self.head_templates:
            print("Used generic")
            content = self.head_templates["generic"].format(mention, relation)
        else:
            content = self.head_templates[relation].format(mention)
        prompt = "{} {}".format(self.system_prompt, content)
        return prompt

    def export(self, config_name: str, path: Path = Path("run_configurations")):
        """Speichert RunConfig mit Namen config_name im Ordner path, default: run_configurations."""
        if not path.exists():
            path.mkdir()
        json_path = path / config_name
        data = {
            "tail_prompt_templates": self.tail_templates,
            "head_prompt_templates": self.head_templates,
            "system_prompt": self.system_prompt,
            "data_type": self.data_type,
            "model_path": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
        }
        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        print("Run-Configuration nach '{}/{}' exportiert".format(path, config_name))

    # Gibt Infos zur RunConfig
    # TODO
    def __str__(self) -> str:
        """Gibt Infos zur RunConfig."""
        s = "System-Prompt: {}\n\n Tail Prompt Templates: {}\n\n Head Prompt Templates: {}\n\n Data: {}\n\n Model-Path: {}\n\n Tensor-Parallel-Size: {}\n".format(
            self.system_prompt,
            self.tail_templates,
            self.head_templates,
            self.data_type,
            self.model_path,
            self.tensor_parallel_size,
        )
        return s

    # Config erstellen aus Daten
    @classmethod
    def from_paths(
        cls,
        prompt_templates_path: str,
        system_prompt_path: str,
        data_type: str,
        model_path: str,
        tensor_parallel_size: int,
    ) -> "RunConfig":
        """Erstellt RunConfig aus Pfaden"""
        templates = load_prompt_templates(prompt_templates_path)
        tail_templates = templates[0]
        head_templates = templates[1]
        check_templates(tail_templates=tail_templates, head_templates=head_templates)
        system_prompt = load_system_prompt(system_prompt_path)
        return RunConfig(
            tail_templates=tail_templates,
            head_templates=head_templates,
            system_prompt=system_prompt,
            data_type=data_type,
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
        )


def check_templates(tail_templates, head_templates):
    """Überprüft prompt templates auf Vollständigkeit"""
    if tail_templates is None:
        raise MissingTemplatesException("Keine Tail Templates in Datei")
    if "generic" not in tail_templates:
        raise MissingGenericException("Keine Generic Template für Tail Completion")
    if head_templates is None:
        raise MissingTemplatesException("Keine Tail Templates in Datei")
    if "generic" not in head_templates:
        raise MissingGenericException("Keine Generic Template für Head Completion")


# Importert run_config mit namen config_name as Ordner path
def import_config(
    config_name: str, path: Path = Path("run_configurations")
) -> "RunConfig":
    if not path.exists:
        print("Kein Ordner, aus dem Importiert werden könnte!")
        return
    json_path = path / config_name
    with open(json_path) as file:
        json_file = json.load(file)
        tail_templates = json_file["tail_prompt_templates"]
        head_templates = json_file["head_prompt_templates"]
        system_prompt = json_file["system_prompt"]
        data_type = json_file["data_type"]
        model_path = json_file["model_path"]
        tensor_parallel_size = json_file["tensor_parallel_size"]
        config = RunConfig(tail_templates, head_templates, system_prompt, data_type)
    return config


# Läd Systemprompt von Pfad
def load_system_prompt(system_prompt_path) -> str:
    with open(system_prompt_path) as file:
        system_prompts = json.load(file)["system"]
        assert isinstance(system_prompts, list)
        system_prompt = "".join(system_prompts)
    return system_prompt


# Läd Prompt-Templates von Pfaden
def load_prompt_templates(prompt_templates_path: str) -> (dict, dict):
    with open(prompt_templates_path) as file:
        json_file = json.load(file)
        tail_templates = json_file["tail"]
        head_templates = json_file["head"]
        assert isinstance(tail_templates, dict) and isinstance(head_templates, dict)

    return tail_templates, head_templates
