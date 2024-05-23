import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from irt2.types import MID, RID, VID
from ktz.collections import path

Tasks = dict[tuple[MID, RID], set[VID]]


from dataclasses import dataclass
from typing import Literal


@dataclass
class Assembler:
    template: str
    system: list[str]
    question: dict[Literal["head", "tail"], dict[str, str]]
    texts: dict[MID, list[str]] | None

    def assemble(
        self,
        direction: Literal["head", "tail"],
        mid: MID,
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

        text = ""
        if self.texts is not None:
            text_lis = self.texts.get(mid, [])
            if len(text_lis):
                text = "\n  - ".join(text_lis)

        prompt = template.format(
            mention=mention,
            relation=relation,
            text=text,
        )

        return prompt

    @classmethod
    def from_paths(
        cls,
        dataset_name: str,
        template_path: str | Path,
        system_path: str | Path,
        question_path: str | Path,
        texts_path: str | Path | None = None,
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

        assert question is not None, "did not find {dataset_name} in {question_path}"

        texts = None
        if texts_path is not None:
            with path(texts_path, is_file=True).open(mode="rb") as fd:
                texts = pickle.load(fd)

        return cls(
            template=template,
            system=system,
            question=question,
            texts=texts,
        )
