import pickle
from dataclasses import dataclass
from itertools import islice
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
    texts: dict[Literal["head", "tail"], dict[tuple[MID, RID] | MID, list[str]]] | None

    def _assemble_text(
        self,
        direction: Literal["head", "tail"],
        mid: MID,
        rid: RID,
        n: int | None = None,
    ) -> str:
        if self.texts is None:
            return ""

        text_dic = self.texts[direction]
        if (mid, rid) in text_dic:
            text_lis = text_dic[(mid, rid)]
        elif mid in text_dic:
            text_lis = text_dic[mid]
        else:
            text_lis = []

        if not len(text_lis):
            return ""

        return " ".join(s.replace("\n", "") for s in islice(text_lis, n))

    def assemble(
        self,
        direction: Literal["head", "tail"],
        mid: MID,
        mention: str,
        rid: RID,
        relation: str,
    ) -> str:
        relation_key = relation.split(":")[1]
        question = self.question[direction][relation_key]

        system = " ".join(self.system)

        template = self.template.format(
            system=system,
            question=question,
        )

        text = self._assemble_text(direction, mid, rid, n=10)

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
        texts_head_path: str | Path | None = None,
        texts_tail_path: str | Path | None = None,
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
        if texts_head_path is not None and texts_tail_path is not None:
            texts = {}
            with path(texts_head_path, is_file=True).open(mode="rb") as fd:
                texts["head"] = pickle.load(fd)
            with path(texts_tail_path, is_file=True).open(mode="rb") as fd:
                texts["tail"] = pickle.load(fd)

        return cls(
            template=template,
            system=system,
            question=question,
            texts=texts,
        )
