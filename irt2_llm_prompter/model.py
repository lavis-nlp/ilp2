from dataclasses import dataclass
from typing import Generator, Iterable

import orjson
import yaml
from irt2.types import Task
from vllm import LLM, SamplingParams


@dataclass
class Prompt:
    task: Task
    mention: str
    relation: str
    body: str


def parse_answer(
    output: str,
    top_k: int | None = None,
) -> list[str]:
    agg = []

    # this successively extracts all json objects from a string
    # exit condition: end of string reached
    while output and (len(agg) < top_k if top_k is not None else True):
        # exit condition: no more {} pairs found
        if "{" not in output or "}" not in output:
            return agg

        start, end = output.index("{"), output.index("}") + 1
        sub, output = output[start:end], output[end:]

        try:
            agg += [s.strip().lower() for s in orjson.loads(sub)["answer"]]

        # malformed json object, continue search
        except (KeyError, orjson.JSONDecodeError):
            continue

    return agg


@dataclass
class Model:
    path: str
    tensor_parallel_size: int
    params: SamplingParams

    llm: LLM | None = None

    def load(self):
        if self.llm is not None:
            return

        self.llm = LLM(
            model=self.path,
            tensor_parallel_size=self.tensor_parallel_size,
        )

    def prompt(self, prompts: Iterable[str]) -> Generator[str, None, None]:
        if self.llm is None:
            self.load()
            assert self.llm is not None

        outputs = self.llm.generate(
            prompts=list(prompts),
            sampling_params=self.params,
        )

        for output in outputs:
            yield output.outputs[0].text

    def parse(self, s: str) -> list[str]:
        return parse_answer(s)
