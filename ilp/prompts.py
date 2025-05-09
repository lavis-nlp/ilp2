import pickle
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import yaml
from irt2.dataset import IRT2
from irt2.types import MID, RID, VID, Split
from ktz.collections import path

Tasks = dict[tuple[MID, RID], set[VID]]


from dataclasses import dataclass
from typing import Literal


@dataclass
class Assembler:
    dataset_name: str
    dataset: IRT2
    mode: Literal[
        "default",
        "prompt-re-ranking",
        "full-re-ranking",
        "ranker-results",
    ]
    split: Split
    template: str
    system: list[str]
    question: dict[Literal["head", "tail"], dict[str, str]]
    texts: dict[Literal["head", "tail"], dict[tuple[MID, RID] | MID, list[str]]] | None
    scores_head: dict[tuple[int, int], np.ndarray] | None
    scores_tail: dict[tuple[int, int], np.ndarray] | None
    n_candidates: int
    mid2idx: dict[int, int] | None
    mentions_per_candidate: int

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

    def get_n_mids_per_candidate(
        self,
        direction: Literal["head", "tail"],
        mid: MID,
        rid: RID,
    ) -> list[set[MID]]:

        top_n_score_vids = self.get_top_n_vids(
            direction=direction,
            task=(mid, rid),
        )

        mid_sets = []

        for vid in top_n_score_vids:
            if self.mentions_per_candidate == 0:
                mid_sets.append(set())
                continue

            mid_set = self.dataset.idmap.vid2mids[Split.train][vid]
            if mid_set is None and self.split == Split.test:
                mid_set = self.dataset.idmap.vid2mids[Split.valid][vid]
                assert mid_set is not None
            mid_sets.append(mid_set)

        return mid_sets

    def get_ranked_vids(
        self,
        direction: Literal["head", "tail"],
        task: tuple[MID, RID],
    ) -> list[int]:
        if "IRT2" in self.dataset_name:
            assert self.mid2idx is not None
            idx = self.mid2idx.get(task[0])
            if idx is None:
                return []
            task = (idx, task[1])

        scores = self._get_scores_for_direction(
            direction=direction,
            task=task,
        )

        if scores is None:
            return []

        if "IRT2" in self.dataset_name:
            scores = np.argsort(scores)[::-1]

        return list(map(int, scores))

    def get_top_n_vids(
        self,
        *args,
        n: int | None = None,
        **kwargs,
    ) -> list[int]:
        if n is None:
            n = self.n_candidates
        return self.get_ranked_vids(*args, **kwargs)[:n]

    def _get_scores_for_direction(
        self,
        direction: Literal["head", "tail"],
        task: tuple[int, int],
    ):
        assert self.scores_head
        assert self.scores_tail

        if direction == "head":
            return self.scores_head.get(task)
        else:
            return self.scores_tail.get(task)

    def assemble(
        self,
        direction: Literal["head", "tail"],
        mid: MID,
        mention: str,
        rid: RID,
        relation: str,
        candidates: str = "",
    ) -> str:
        relation_key = relation.split(":")[1]
        question = self.question[direction][relation_key]

        system = " ".join(self.system)

        if self.n_candidates > 0 and candidates == "":
            mid_sets = self.get_n_mids_per_candidate(direction, mid, rid)

            if self.mode == "default":
                candidates = ", ".join(
                    self.dataset.idmap.mid2str[mid]
                    for mid_set in mid_sets
                    for mid in list(mid_set)[: self.mentions_per_candidate]
                )
            else:
                top_n_vids = self.get_top_n_vids(direction=direction, task=(mid, rid))

                top_n_entity_names: list[str] = [
                    self.dataset.idmap.vid2str[vid].split(":")[1] for vid in top_n_vids
                ]

                candidates = "\n".join(
                    f"{i}: {top_n_entity_names[i]}, {', '.join(self.dataset.idmap.mid2str[mid] for mid in list(mid_set)[:self.mentions_per_candidate])}"
                    for i, mid_set in enumerate(mid_sets)
                )

                candidates.replace(",\n", "\n")

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
        dataset: IRT2,
        mode: Literal[
            "default",
            "prompt-re-ranking",
            "full-re-ranking",
            "ranker-results",
        ],
        split_str: str,
        template_path: str | Path,
        system_path: str | Path,
        question_path: str | Path,
        texts_head_path: str | Path | None = None,
        texts_tail_path: str | Path | None = None,
        n_candidates: int = 0,
        mentions_per_candidate: int = 1,
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

            scores_head_dict: dict[tuple[int, int], np.ndarray] | None = None
            scores_tail_dict: dict[tuple[int, int], np.ndarray] | None = None

            mid2idx = None

            if n_candidates > 0:

                if "IRT2" in dataset_name:
                    scores_path = next(
                        dataset.path.glob(f"*scores.{split_str}.h5"),
                        None,
                    )
                    mid2idx_path = next(
                        dataset.path.glob("mid2idx-irt2-*.pkl"),
                        None,
                    )

                    if mid2idx_path:
                        with open(mid2idx_path, "rb") as file:
                            mid2idx = pickle.load(file)

                    assert scores_path != None

                    with h5py.File(scores_path, "r") as scores_fd:
                        head_tasks = scores_fd["head"]["tasks"]  # type: ignore
                        head_scores = scores_fd["head"]["scores"]  # type: ignore
                        scores_head_dict = {
                            (task[0], task[1]): head_scores[i]  # type: ignore
                            for i, task in enumerate(head_tasks)  # type: ignore
                        }
                        tail_tasks = scores_fd["tail"]["tasks"]  # type: ignore
                        tail_scores = scores_fd["tail"]["scores"]  # type: ignore
                        scores_tail_dict = {
                            (task[0], task[1]): tail_scores[i]  # type: ignore
                            for i, task in enumerate(tail_tasks)  # type: ignore
                        }

                else:

                    prefix = dataset_name.split("/")[1]

                    scores_path = next(
                        dataset.path.glob(f"{prefix}-{split_str}.pkl"), None
                    )
                    assert scores_path != None

                    with open(scores_path, "rb") as file:
                        scores = pickle.load(file)

                        scores_head_dict = scores["head predictions"]
                        scores_tail_dict = scores["tail predictions"]

        assert question is not None, "did not find {dataset_name} in {question_path}"

        texts = None
        if texts_head_path is not None and texts_tail_path is not None:
            texts = {}
            with path(texts_head_path, is_file=True).open(mode="rb") as fd:
                texts["head"] = pickle.load(fd)
            with path(texts_tail_path, is_file=True).open(mode="rb") as fd:
                texts["tail"] = pickle.load(fd)

        if split_str == "test":
            split = Split.test
        else:
            split = Split.valid

        return cls(
            dataset_name=dataset_name,
            dataset=dataset,
            mode=mode,
            split=split,
            template=template,
            system=system,
            question=question,
            texts=texts,
            scores_head=scores_head_dict,
            scores_tail=scores_tail_dict,
            n_candidates=n_candidates,
            mid2idx=mid2idx,
            mentions_per_candidate=mentions_per_candidate,
        )
