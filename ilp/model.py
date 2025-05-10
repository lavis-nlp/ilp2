import re
from dataclasses import dataclass
from typing import Generator, Iterable, Literal

import orjson
import torch
from vllm import LLM
from vllm.sampling_params import BeamSearchParams, SamplingParams

import ilp
from ilp.config import Config


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
    assert top_k is None, "unused?"

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

    def load(self):
        raise NotImplementedError

    def prompt(
        self,
        config: Config,
        prompts: Iterable[str],
    ) -> Generator[str, None, None]:
        raise NotImplementedError

    def parse(
        self,
        answer: str,
    ) -> list[str]:
        match self.parser:
            case "json":
                return parse_json_answer(answer)
            case "csv":
                return parse_csv_answer(answer)


class VLLMModel(ModelBase):
    llm: LLM | None = None

    tensor_parallel_size: int
    gpu_memory_utilization: float
    quantization: str | None
    use_beam_search: bool
    max_tokens: int

    def __init__(
        self,
        path: str,
        # vllm
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        quantization: str | None,
        use_beam_search: bool,
        max_tokens: int,
        # ilp
        parser: Literal["json", "csv"],
    ):
        super().__init__(path, tensor_parallel_size, parser)

        self.use_beam_search = use_beam_search
        self.gpu_memory_utilization = gpu_memory_utilization
        self.quantization = quantization
        self.max_tokens = max_tokens

    def load(self):
        if self.llm is not None:
            return

        ilp.console.log(f"Loading vLLM model from {self.path}")

        self.llm = LLM(
            model=self.path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            quantization=self.quantization,
            enable_prefix_caching=True,
            max_model_len=self.max_tokens,
        )

        ilp.console.log("Finished loading vLLM model")

    def _prompt_beam_search(self, config: Config, prompts):
        assert self.llm
        assert self.use_beam_search

        params = BeamSearchParams(
            beam_width=config.sampling_beam_width,
            max_tokens=config.model_max_tokens,
            length_penalty=config.sampling_repetition_penalty,
        )

        outputs = self.llm.beam_search(
            prompts=[{"prompt": p} for p in prompts],
            params=params,
            # disabled by default in the current version
            # use_tqdm=not ilp.debug,
        )

        tokenizer = self.llm.get_tokenizer()
        for prompt, output in zip(prompts, outputs):
            # unfortunately, beam_search() outputs the
            # input prompt plus special tokens ---
            # so we require the encoded prompt (which contains
            # special tokens) and cut it out
            tokens = output.sequences[0].tokens[len(tokenizer.encode(prompt)) :]
            yield tokenizer.decode(tokens)

    def _prompt_random_sampling(self, config: Config, prompts):
        assert not self.use_beam_search

        params = SamplingParams(
            temperature=config.sampling_temperature,
            top_p=config.sampling_top_p,
            max_tokens=config.model_max_tokens,
            repetition_penalty=config.sampling_repetition_penalty,
        )

        # prompts is "deprected" and we should use "inputs" instead
        # however, there is no input parameter (???)
        outputs = self.llm.generate(
            prompts=prompts,
            sampling_params=params,
            use_tqdm=False,
        )

        for output in outputs:
            yield output.outputs[0].text

    def _prompt_post(self, results):
        # TODO hacky solution, we should define supported
        # model families and their output formats somewhere

        spath = str(self.path).lower()

        for result in results:
            # LLAMA models require no other postprocessing
            if "meta-llama-3-70b-instruct" in spath:
                yield result.strip()
                continue

            if "deepseek-r1-distill-llama-70b" in spath:
                try:
                    yield result.split("</think>")[1].strip()
                except IndexError:
                    yield result.strip()
                continue

            assert False, f"not supported: {spath}"

    def prompt(
        self,
        config: Config,
        prompts: Iterable[str],
    ) -> Generator[str, None, None]:
        if self.llm is None:
            self.load()
            assert self.llm is not None

        if self.use_beam_search:
            gen = self._prompt_beam_search(
                config,
                list(prompts),
            )

        else:
            gen = self._prompt_random_sampling(
                config,
                list(prompts),
            )

        yield from self._prompt_post(gen)


# currently unused
# class HuggingFaceModel(ModelBase):

#     model_kwargs = None
#     dtype: torch.dtype
#     batch_size: int
#     model = None
#     tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast

#     def __init__(
#         self,
#         path: str,
#         tensor_parallel_size: int,
#         parser: Literal["json", "csv"],
#         model_kwargs,
#         dtype: torch.dtype,
#         batch_size: int,
#     ):
#         assert False, "adjust to new config"
#         super().__init__(path, tensor_parallel_size, parser)
#         self.model_kwargs = model_kwargs
#         self.dtype = dtype
#         self.batch_size = batch_size

#     @classmethod
#     def from_config(cls, config):

#         model_kwargs = {
#             "max_new_tokens": config.max_tokens,
#             "top_p": config.top_p,
#             "repetition_penalty": config.repetition_penalty,
#         }

#         return cls(
#             path=str(config.model_path),
#             tensor_parallel_size=config.tensor_parallel_size,
#             parser=config.parser,
#             model_kwargs=model_kwargs,
#             dtype=getattr(torch, config.dtype),
#             batch_size=config.batch_size,
#         )

#     def load(self):

#         if self.model is not None:
#             return

#         ilp.console.log(f"Loading huggingface model from {self.path}")

#         self.tokenizer = AutoTokenizer.from_pretrained(self.path, padding_side="left")
#         self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.path, torch_dtype=self.dtype, device_map="auto"
#         )

#         ilp.console.log("Finished loading huggingface model")

#     def prompt(self, prompts: Iterable[str]) -> Generator[str, None, None]:
#         if self.model is None:
#             self.load()
#             assert self.model is not None
#             assert self.tokenizer is not None

#         terminators = [
#             self.tokenizer.eos_token_id,
#             self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
#         ]

#         promptlist = [[{"role": "system", "content": prompt}] for prompt in prompts]
#         assert len(promptlist)

#         prompt_batches = [
#             promptlist[i : i + self.batch_size]
#             for i in range(0, len(promptlist), self.batch_size)
#         ]

#         device = self.model.device

#         i = 0
#         for batch in prompt_batches:
#             i += 1

#             print("Start batch {} of {}".format(i, len(prompt_batches)))

#             texts = self.tokenizer.apply_chat_template(
#                 batch, add_generation_prompt=True, tokenize=False
#             )

#             inputs = self.tokenizer(texts, padding="longest", return_tensors="pt", add_special_tokens=False)  # type: ignore
#             inputs = {key: val.to(device) for key, val in inputs.items()}
#             temp_texts = self.tokenizer.batch_decode(
#                 inputs["input_ids"], skip_special_tokens=True
#             )

#             gen_tokens = self.model.generate(
#                 **inputs,
#                 pad_token_id=self.tokenizer.eos_token_id,
#                 eos_token_id=terminators,
#                 **(self.model_kwargs or {}),
#             )

#             outputs = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
#             gen_texts = [i[len(temp_texts[idx]) :] for idx, i in enumerate(outputs)]

#             if gen_texts is None:
#                 return

#             for output in gen_texts:
#                 yield output
