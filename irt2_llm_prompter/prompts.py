from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import orjson
import yaml
from irt2.dataset import IRT2
from irt2.types import MID, RID, VID
from ktz.collections import path

Tasks = dict[tuple[MID, RID], set[VID]]


from dataclasses import dataclass
from typing import Literal

from irt2.types import Task


@dataclass
class Assembler:
    template: str
    system: list[str]
    question: dict[Literal["head", "tail"], dict[str, str]]

    def assemble(
        self,
        direction: Literal["head", "tail"],
        mention: str,
        relation: str,
    ) -> str:
        relation_key = relation.split(":")[1]
        question = self.question[direction][relation_key]

        system = " ".join(self.system)

        template = self.template.format(
            system=system,
            question=question,
        )
        prompt = template.format(
            mention=mention,
            relation=relation,
        )

        return prompt

    @classmethod
    def from_paths(
        cls,
        dataset_name: str,
        template_path: str | Path,
        system_path: str | Path,
        question_path: str | Path,
    ):
        with (
            path(template_path, is_file=True).open(mode="r") as tmpl_fd,
            path(system_path, is_file=True).open(mode="r") as sys_fd,
            path(question_path, is_file=True).open(mode="r") as q_fd,
        ):
            template = tmpl_fd.read()
            system = yaml.safe_load(sys_fd)

            question = None
            for conf in yaml.safe_load(q_fd):
                if dataset_name not in conf["datasets"]:
                    continue
                question = conf["prompts"]

            assert (
                question is not None
            ), "did not find {dataset_name} in {question_path}"

            return cls(
                template=template,
                system=system,
                question=question,
            )
