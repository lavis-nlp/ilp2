from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import yaml
from irt2.types import MID, RID, VID
from ktz.collections import path

import ilp

Tasks = dict[tuple[MID, RID], set[VID]]


@dataclass
class Config:
    # mode configuration
    mode: Literal[
        "default",
        "prompt-re-ranking",
        "full-re-ranking",
        "ranker-results",
    ]

    # data configuration
    dataset_config: str
    dataset_key: str
    dataset_split: Literal["validation", "test"]
    dataset_limit_tasks: int | None
    dataset_texts_head: str | None
    dataset_texts_tail: str | None

    # model configuration
    model_path: str
    model_parser: Literal["json", "csv"]
    model_engine: Literal["vllm", "huggingface"]
    model_quantization: Literal["fp8"] | None
    model_max_tokens: int
    model_tensor_parallel_size: int
    model_use_beam_search: bool

    # vllm model params
    model_gpu_memory_utilization: float

    # prompt templates
    prompt_template: str  # conf/prompts/template
    prompt_system: str  # conf/prompts/system
    prompt_question: str  # conf/prompts/question

    # preprocessing/processing
    stopwords_path: str | None  # conf/stopwords
    use_stemmer: bool
    n_candidates: int  # top n candidates given to the model
    mentions_per_candidate: int  # mentions per candidate proposed to the model
    give_true_candidates: bool  # TODO doc

    # sampling params (beam search)
    sampling_beam_width: int  # gets expensive fast
    sampling_length_penalty: float

    # sampling params (random sampling)
    # if use_beam_search is False
    # must be False for random sampling:
    sampling_early_stopping: bool
    # consider tokens until their cum. prob. reaches:
    sampling_top_p: float
    # penalize new tokens if they appeared before:
    sampling_repetition_penalty: float

    # sampling params (shared)
    sampling_temperature: float

    # --- persistence

    def save(self, to: Path | str):
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
        sep = "\n  - "
        return "Config:" + sep + sep.join(f"{k}: {v}" for k, v in asdict(self).items())
