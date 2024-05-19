from typing import Generator, Iterable

from vllm import LLM, SamplingParams


class Model:
    model_path: str
    tensor_parallel_size: int
    params: SamplingParams
    is_loaded: bool = False

    llm: LLM

    def __init__(
        self,
        path: str,
        params: SamplingParams,
        tensor_parallel_size: int,
    ):
        self.model_path = path
        self.tensor_parallel_size = tensor_parallel_size
        self.params = params

    def load_model(self):
        """Läd model aus Objektconfig"""
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
        )

        self.is_loaded = True

    def prompt(self, prompts: Iterable[str]) -> Generator[str, None, None]:
        outputs = self.llm.generate(
            prompts=list(prompts),
            sampling_params=self.params,
        )

        for output in outputs:
            yield output.outputs[0].text

    # def prompt_model(self, prompt: str = "", prompts: List = []) -> List[RequestOutput]:
    #     """Promptet model mit prompt"""
    #     if not self.is_loaded:
    #         self.load_model()

    #     if len(prompts) == 0:
    #         prompts = [
    #             prompt,
    #         ]
    #     result = self.llm.generate(
    #         prompts=prompts, sampling_params=self.params, use_tqdm=False
    #     )
    #     return result

    # def set_sampling_params(self, sampling_params: SamplingParams):
    #     self.params = sampling_params

    # def parse_single_answer(result: List[RequestOutput]):
    #     return result[0].outputs.text
