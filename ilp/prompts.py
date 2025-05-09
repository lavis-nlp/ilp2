import pickle
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Literal, Sequence

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
    scores_head: dict[tuple[int, int], Sequence[VID]] | None
    scores_tail: dict[tuple[int, int], Sequence[VID]] | None
    n_candidates: int
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
    ) -> Sequence[VID]:
        scores = self._get_scores_for_direction(
            direction=direction,
            task=task,
        )

        if scores is None:
            return []

        return scores

    def get_top_n_vids(
        self,
        *args,
        n: int | None = None,
        **kwargs,
    ) -> Sequence[VID]:
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
                top_n_vids = self.get_top_n_vids(
                    direction=direction,
                    task=(mid, rid),
                )

                top_n_entity_names: list[str] = [
                    self.dataset.idmap.vid2str[vid].split(":")[1] for vid in top_n_vids
                ]

                candidates = ""
                for i, mid_set in enumerate(mid_sets):
                    mentions = f"{top_n_entity_names[i]},"
                    mentions += ", ".join(
                        self.dataset.idmap.mid2str[mid]
                        for mid in list(mid_set)[: self.mentions_per_candidate]
                    )
                    candidates += f"{i}: {mentions}"

        candidates.replace(",\n", "\n")
        breakpoint()

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

    @staticmethod
    def _load_scores_irt(
        dataset: IRT2,
        split_str: str,
    ):
        heads: dict[tuple[int, int], Sequence[VID]] | None = None
        tails: dict[tuple[int, int], Sequence[VID]] | None = None

        scores_fname = f"*scores.{split_str}.h5"
        scores_path = next(dataset.path.glob(f"*scores.{split_str}.h5"), None)
        assert scores_path != None, scores_fname

        mid2idx_fname = "mid2idx-irt2-*.pkl"
        mid2idx_path = next(dataset.path.glob(mid2idx_fname), None)
        assert mid2idx_path != None, mid2idx_fname
        with mid2idx_path.open(mode="rb") as file:
            mid2idx = pickle.load(file)

        def to_vids(scores):
            vids = np.argsort(scores)[::-1]
            return list(map(int, vids))

        with h5py.File(scores_path, "r") as scores_fd:
            head_tasks = scores_fd["head"]["tasks"]  # type: ignore
            head_scores = scores_fd["head"]["scores"]  # type: ignore

            heads = {
                (mid2idx.get(task[0]), task[1]): to_vids(head_scores[i])  # type: ignore
                for i, task in enumerate(head_tasks)  # type: ignore
            }

            tail_tasks = scores_fd["tail"]["tasks"]  # type: ignore
            tail_scores = scores_fd["tail"]["scores"]  # type: ignore

            tails = {
                (task[0], task[1]): to_vids(tail_scores[i])  # type: ignore
                for i, task in enumerate(tail_tasks)  # type: ignore
            }

        return heads, tails

    @staticmethod
    def _load_scores_blp(
        dataset: IRT2,
        dataset_name: str,
        split_str: str,
    ):
        prefix = dataset_name.split("/")[1]

        scores_fname = f"{prefix}-{split_str}.pkl"
        scores_path = next(dataset.path.glob(scores_fname), None)
        assert scores_path != None, scores_fname

        with open(scores_path, "rb") as file:
            scores = pickle.load(file)
            scores_head = scores["head predictions"]
            scores_tail = scores["tail predictions"]

        return scores_head, scores_tail

    @classmethod
    def _load_scores(
        cls,
        dataset: IRT2,
        dataset_name: str,
        split_str: str,
    ):
        scores_head: dict[tuple[int, int], Sequence[VID]]
        scores_tail: dict[tuple[int, int], Sequence[VID]]

        if "IRT2" in dataset_name:
            scores_head, scores_tail = cls._load_scores_irt(
                dataset=dataset,
                split_str=split_str,
            )

        elif "BLP" in dataset_name:
            scores_head, scores_tail = cls._load_scores_blp(
                dataset=dataset,
                dataset_name=dataset_name,
                split_str=split_str,
            )

        else:
            assert False, f"unknown dataset: {dataset_name}"

        if split_str == "validation":
            assert set(scores_head) == set(dataset.open_kgc_val_heads)
            assert set(scores_tail) == set(dataset.open_kgc_val_tails)

        if split_str == "test":
            assert set(scores_head) == set(dataset.open_kgc_test_heads)
            assert set(scores_tail) == set(dataset.open_kgc_test_tails)

        return scores_head, scores_tail

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

            scores_head, scores_tail = None, None
            if n_candidates > 0:
                scores_head, scores_tail = cls._load_scores(
                    dataset,
                    dataset_name,
                    split_str,
                )

        assert question is not None, f"did not find {dataset_name} in {question_path}"

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
            scores_head=scores_head,
            scores_tail=scores_tail,
            n_candidates=n_candidates,
            mentions_per_candidate=mentions_per_candidate,
        )
