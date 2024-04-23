import json
from pathlib import Path


# @dataclass
# class RunConfig:
class run_config:
    irt2_data_path: str

    tail_templates: dict
    head_templates: dict
    system_prompt: str

    # data:irt2.dataset.IRT2

    def __init__(
        self,
        tail_templates: dict,
        head_templates: dict,
        system_prompt: str,
        irt2_data_path: str,
    ):
        self.tail_templates = tail_templates
        self.head_templates = head_templates
        self.system_prompt = system_prompt
        self.irt2_data_path = irt2_data_path
        # self.data = IRT2.from_dir(path=irt2_data_path)

    # Gibt fertigen tail-completion-prompt zurück
    def get_tail_prompt(self, mention: str, relation: str) -> str:
        prompt = "{} {}".format(
            self.system_prompt, self.tail_templates[relation].format(mention)
        )
        return prompt

    # Gibt fertigen head-completion-prompt zurück
    def get_head_prompt(self, mention: str, relation: str) -> str:
        prompt = "{} {}".format(
            self.system_prompt, self.head_templates[relation].format(mention)
        )
        return prompt

    def export(self, config_name: str, path: Path = Path("run_configurations")):
        """Speichert run_config mit Namen config_name im Ordner path, default: run_configurations."""
        if not path.exists():
            path.mkdir()
        json_path = path / config_name
        data = {
            "tail_prompt_templates": self.tail_templates,
            "head_prompt_templates": self.head_templates,
            "system_prompt": self.system_prompt,
            "irt2_data_path": self.irt2_data_path,
        }
        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        print("Run-Configuration nach '{}/{}' exportiert".format(path, config_name))

    # Gibt Infos zur run_config
    # TODO
    def __str__(self) -> str:
        """Gibt Infos zur run_config."""
        s = "-" * 20 + "CONFIG" + "-" * 20 + "\n"
        s += "System-Prompt: {}\n\n Tail Prompt Templates: {}\n\n Head Prompt Templates: {}\n\n IRT2-Data: {}\n".format(
            self.system_prompt,
            self.tail_templates,
            self.head_templates,
            self.irt2_data_path,
        )
        s += "-" * 46
        return s

    # Config erstellen aus Daten
    @classmethod
    def from_paths(
        cls,
        prompt_templates_path: str,
        system_prompt_path: str,
        irt2_data_path: str,
    ) -> "run_config":
        templates = load_prompt_templates(prompt_templates_path)
        tail_templates = templates[0]
        head_templates = templates[1]
        system_prompt = load_system_prompt(system_prompt_path)
        return run_config(
            tail_templates,
            head_templates,
            system_prompt,
            irt2_data_path,
        )



# Importert run_config mit namen config_name as Ordner path
def import_config(
    config_name: str, path: Path = Path("run_configurations")
) -> run_config:
    if not path.exists:
        print("Kein Ordner, aus dem Importiert werden könnte!")
        return
    json_path = path / config_name
    with open(json_path) as file:
        json_file = json.load(file)
        tail_templates = json_file["tail_prompt_templates"]
        head_templates = json_file["head_prompt_templates"]
        system_prompt = json_file["system_prompt"]
        irt2_data_path = json_file["irt2_data_path"]
        config = run_config(
            tail_templates, head_templates, system_prompt, irt2_data_path
        )
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
