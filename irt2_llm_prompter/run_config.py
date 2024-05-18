import json
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from ktz.collections import path

import irt2_llm_prompter as ilp
from irt2_llm_prompter.customErrors import (
    MissingGenericException,
    MissingTemplatesException,
)


@dataclass
class RunConfig:
    # prompts
    tail_templates: dict
    head_templates: dict
    system_prompt: str

    # model configuration
    model_path: str
    tensor_parallel_size: int

    # meta information

    system_prompt_path: str | None = None
    question_prompt_path: str | None = None

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

    # Config erstellen aus Daten
    @classmethod
    def from_paths(
        cls,
        prompt_templates_path: Path,
        system_prompt_path: Path,
        model_path: Path,
        tensor_parallel_size: int,
    ) -> "RunConfig":
        """Erstellt RunConfig aus Pfaden"""
        tail_templates, head_templates = load_prompt_templates(prompt_templates_path)

        check_templates(
            tail_templates=tail_templates,
            head_templates=head_templates,
        )

        system_prompt = load_system_prompt(system_prompt_path)

        return cls(
            tail_templates=tail_templates,
            head_templates=head_templates,
            system_prompt=system_prompt,
            model_path=str(model_path),
            tensor_parallel_size=tensor_parallel_size,
        )

    # --- persistence

    def save(self, to: Path | str):
        """Speichert RunConfig mit Namen config_name im Ordner path, default: run_configurations."""
        out = path(to)
        out.parent.mkdir(exist_ok=True, parents=True)

        with out.open("w") as fd:
            yaml.safe_dump(asdict(self), fd)

        ilp.console.log(f"exported run config to {out}")

    @classmethod
    def load(cls, fname: Path | str):
        with path(fname, is_file=True).open(mode="r") as fd:
            return cls(**yaml.safe_load(fd))

    def __str__(self) -> str:
        """Gibt Infos zur RunConfig."""
        rep = f"""
        run configuration:
          - {len(self.tail_templates)} tail templates
          - {len(self.head_templates)} head templates
          - system prompt: {self.system_prompt_path}
          - question prompts: {self.question_prompt_path}
          - model path: {self.model_path} (tps={self.tensor_parallel_size})
        """

        return textwrap.dedent(rep).strip()


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


# Läd Systemprompt von Pfad
def load_system_prompt(system_prompt_path) -> str:
    with open(system_prompt_path) as file:
        system_prompts = json.load(file)["system"]
        assert isinstance(system_prompts, list)
        system_prompt = "".join(system_prompts)
    return system_prompt


# Läd Prompt-Templates von Pfaden
def load_prompt_templates(prompt_templates_path: Path) -> tuple[dict, dict]:
    with open(prompt_templates_path) as file:
        json_file = json.load(file)
        tail_templates = json_file["tail"]
        head_templates = json_file["head"]
        assert isinstance(tail_templates, dict) and isinstance(head_templates, dict)

    return tail_templates, head_templates
