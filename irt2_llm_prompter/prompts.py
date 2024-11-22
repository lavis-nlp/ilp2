import pickle
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import yaml
from irt2.dataset import IRT2
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
    scores_head: dict[np.int64, np.ndarray] | None
    scores_tail: dict[np.int64, np.ndarray] | None
    n_candidates: int

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
        dataset: IRT2,
    ) -> str:
        relation_key = relation.split(":")[1]
        question = self.question[direction][relation_key]

        system = " ".join(self.system)

        candidates = ""

        if (
            self.n_candidates > 0
            and self.scores_head is not None
            and self.scores_tail is not None
        ):
            if direction == "head":
                scores = self.scores_head.get((mid, rid))
            else:
                scores = self.scores_tail.get((mid, rid))
            top_n_scores = np.argsort(scores)[::-1][: self.n_candidates]
            top_n_candidates = [
                dataset.idmap.vid2str[vid].split(":")[1] for vid in top_n_scores
            ]
            candidates = "This is a list of possible candidates: " + ", ".join(
                top_n_candidates
            )

        template = self.template.format(
            system=system,
            question=question,
            candidates=candidates,
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
        scores_path: str | Path | None = None,
        n_candidates: int = 0,
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

            scores_head_dict = None
            scores_tail_dict = None

            if n_candidates > 0 and scores_path != None:
                with h5py.File(scores_path, "r") as scores_fd:
                    head_tasks = scores_fd["head"]["tasks"]
                    head_scores = scores_fd["head"]["scores"]
                    scores_head_dict = {
                        tuple(task): head_scores[i] for i, task in enumerate(head_tasks)
                    }
                    tail_tasks = scores_fd["tail"]["tasks"]
                    tail_scores = scores_fd["tail"]["scores"]
                    scores_tail_dict = {
                        tuple(task): tail_scores[i] for i, task in enumerate(tail_tasks)
                    }

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
            scores_head=scores_head_dict,
            scores_tail=scores_tail_dict,
            n_candidates=n_candidates,
        )
