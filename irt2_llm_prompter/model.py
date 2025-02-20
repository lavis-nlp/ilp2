from dataclasses import dataclass
from typing import Generator, Iterable, Literal

import orjson
from vllm import LLM, SamplingParams

import irt2_llm_prompter as ilp


def normalize(lis: list[str]):
    return [s.strip().lower() for s in lis]


def parse_json_answer(
    output: str,
    top_k: int | None = None,
) -> list[str]:
    agg: list[str] = []

    # this successively extracts all json objects from a string
    # exit condition: end of string reached
    while output and (len(agg) < top_k if top_k is not None else True):
        # exit condition: no more {} pairs found
        if "{" not in output or "}" not in output:
            return normalize(agg)

        start, end = output.index("{"), output.index("}") + 1
        sub, output = output[start:end], output[end:]

        try:
            raw = orjson.loads(sub)["answer"]

            if isinstance(raw, str):
                agg.append(raw)
            elif isinstance(raw, list):
                agg += [str(x) for x in raw]

        # malformed json object, continue search
        except (KeyError, orjson.JSONDecodeError):
            continue

    return normalize(agg)


def parse_csv_answer(
    output: str,
    top_k: int | None = None,
) -> list[str]:
    agg: list[str] = output.split(",")
    return normalize(agg)


@dataclass
class Model:
    path: str
    tensor_parallel_size: int
    parser: Literal["json", "csv"]
    params: SamplingParams

    llm: LLM | None = None

    def load(self):
        if self.llm is not None:
            return

        ilp.console.log(f"loading model from {self.path}")

        self.llm = LLM(
            model=self.path,
            tensor_parallel_size=self.tensor_parallel_size,
        )

        ilp.console.log(f"finished loading model")

    def prompt(self, prompts: Iterable[str]) -> Generator[str, None, None]:
        if self.llm is None:
            self.load()
            assert self.llm is not None

        promptlist = list(prompts)
        assert len(promptlist)

        outputs = self.llm.generate(
            prompts=promptlist,
            sampling_params=self.params,
            use_tqdm=not ilp.debug,
        )

        for output in outputs:
            yield output.outputs[0].text

    def parse(self, s: str) -> list[str]:
        match self.parser:
            case "json":
                return parse_json_answer(s)
            case "csv":
                return parse_csv_answer(s)
            case _:
                return parse_csv_answer(s)

    @classmethod
    def from_config(cls, config):
        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            use_beam_search=config.use_beam_search,
            best_of=config.best_of,
            max_tokens=config.max_tokens,
            repetition_penalty=config.repetition_penalty,
        )

        return cls(
            path=str(config.model_path),
            params=sampling_params,
            parser=config.parser,
            tensor_parallel_size=config.tensor_parallel_size,
        )
