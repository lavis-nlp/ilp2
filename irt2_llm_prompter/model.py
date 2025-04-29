from csv import Error
from dataclasses import dataclass
from pyexpat import model
from typing import Generator, Iterable, Literal

import re

from click import prompt
import orjson
from traitlets import default
from vllm import LLM, SamplingParams

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
import torch

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
    if output.count(":") == 1:
        output = output.split(":")[1]
    agg: list[str] = re.split(r"[,:\n]", output)
    # agg: list[str] = output.split(",")
    return normalize(agg)


@dataclass
class ModelBase:
    path: str
    tensor_parallel_size: int
    parser: Literal["json", "csv"]

    @classmethod
    def from_config(cls, config):
        match config.engine:
            case "huggingface":
                return HuggingFaceModel.from_config(config)
            case "vllm":
                return VLLMModel.from_config(config)
        raise RuntimeError

    def load(self):
        raise NotImplementedError

    def prompt(self, prompts: Iterable[str]) -> Generator[str, None, None]:
        raise NotImplementedError

    def parse(self, s: str) -> list[str]:
        match self.parser:
            case "json":
                return parse_json_answer(s)
            case "csv":
                return parse_csv_answer(s)
            case _:
                return parse_csv_answer(s)


class VLLMModel(ModelBase):
    llm: LLM | None = None
    params: SamplingParams

    def __init__(
        self,
        path: str,
        tensor_parallel_size: int,
        parser: Literal["json", "csv"],
        sampling_params: SamplingParams,
    ):
        super().__init__(path, tensor_parallel_size, parser)
        self.params = sampling_params

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
            tensor_parallel_size=config.tensor_parallel_size,
            parser=config.parser,
            sampling_params=sampling_params,
        )

    def load(self):
        if self.llm is not None:
            return

        ilp.console.log(f"Loading vLLM model from {self.path}")

        self.llm = LLM(
            model=self.path,
            tensor_parallel_size=self.tensor_parallel_size,
        )

        ilp.console.log("Finished loading vLLM model")

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


class HuggingFaceModel(ModelBase):

    model_kwargs = None
    dtype: torch.dtype
    batch_size: int
    model = None
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast

    def __init__(
        self,
        path: str,
        tensor_parallel_size: int,
        parser: Literal["json", "csv"],
        model_kwargs,
        dtype: torch.dtype,
        batch_size: int,
    ):
        super().__init__(path, tensor_parallel_size, parser)
        self.model_kwargs = model_kwargs
        self.dtype = dtype
        self.batch_size = batch_size

    @classmethod
    def from_config(cls, config):

        model_kwargs = {
            "max_new_tokens": config.max_tokens,
            "do_sample": not config.use_beam_search,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
        }

        if model_kwargs["do_sample"]:
            model_kwargs["temperature"] = config.temperature
        else:
            model_kwargs["num_beams"] = config.best_of
            model_kwargs["temperature"] = None

        return cls(
            path=str(config.model_path),
            tensor_parallel_size=config.tensor_parallel_size,
            parser=config.parser,
            model_kwargs=model_kwargs,
            dtype=getattr(torch, config.dtype),
            batch_size=config.batch_size,
        )

    def load(self):

        if self.model is not None:
            return

        ilp.console.log(f"Loading huggingface model from {self.path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.path, padding_side="left")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.path, torch_dtype=self.dtype, device_map="auto"
        )

        ilp.console.log("Finished loading huggingface model")

    def prompt(self, prompts: Iterable[str]) -> Generator[str, None, None]:
        if self.model is None:
            self.load()
            assert self.model is not None
            assert self.tokenizer is not None

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        promptlist = [[{"role": "system", "content": prompt}] for prompt in prompts]
        assert len(promptlist)

        prompt_batches = [
            promptlist[i : i + self.batch_size]
            for i in range(0, len(promptlist), self.batch_size)
        ]

        device = self.model.device

        i = 0
        for batch in prompt_batches:
            i += 1

            print("Start batch {} of {}".format(i, len(prompt_batches)))

            texts = self.tokenizer.apply_chat_template(
                batch, add_generation_prompt=True, tokenize=False
            )

            inputs = self.tokenizer(texts, padding="longest", return_tensors="pt", add_special_tokens=False)  # type: ignore
            inputs = {key: val.to(device) for key, val in inputs.items()}
            temp_texts = self.tokenizer.batch_decode(
                inputs["input_ids"], skip_special_tokens=True
            )

            gen_tokens = self.model.generate(
                **inputs,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=terminators,
                **(self.model_kwargs or {}),
            )

            outputs = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            gen_texts = [i[len(temp_texts[idx]) :] for idx, i in enumerate(outputs)]

            if gen_texts is None:
                return

            for output in gen_texts:
                yield output
