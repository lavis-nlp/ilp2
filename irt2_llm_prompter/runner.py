import gzip
import json
import pickle
import textwrap
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import islice
from pathlib import Path
from traceback import print_exc
from typing import Generator, Iterable, List, Literal

import orjson
import yaml
from irt2.dataset import IRT2
from irt2.evaluation import Predictions, evaluate
from irt2.types import MID, RID, VID, Split, Task
from ktz.collections import path
from vllm import RequestOutput, SamplingParams

import irt2_llm_prompter as ilp
from irt2_llm_prompter import model_prompter
from irt2_llm_prompter.model_prompter import Model

Tasks = dict[tuple[MID, RID], set[VID]]


# TODO usage?
# def check_templates(tail_templates, head_templates):
#     """Überprüft prompt templates auf Vollständigkeit"""
#     if tail_templates is None:
#         raise MissingTemplatesException("Keine Tail Templates in Datei")
#     if "generic" not in tail_templates:
#         raise MissingGenericException("Keine Generic Template für Tail Completion")
#     if head_templates is None:
#         raise MissingTemplatesException("Keine Tail Templates in Datei")
#     if "generic" not in head_templates:
#         raise MissingGenericException("Keine Generic Template für Head Completion")


def parse_answer(output: str) -> list[str]:
    # this successively extracts all json objects from a string

    agg = []

    # exit condition: end of string reached
    while output:
        # exit condition: no more {} pairs found
        if "{" not in output or "}" not in output:
            return agg

        start, end = output.index("{"), output.index("}") + 1
        sub, output = output[start:end], output[end:]

        try:
            agg += [s.strip().lower() for s in orjson.loads(sub)["answer"]]

        # malformed json object, continue search
        except (KeyError, json.JSONDecodeError):
            continue

    return agg


def load_system_prompt(fpath: str | Path) -> str:
    with path(fpath, is_file=True).open(mode="r") as fd:
        system_prompts = json.load(fd)["system"]

        assert isinstance(system_prompts, list)
        system_prompt = "".join(system_prompts)

    return system_prompt


def load_prompt_templates(fpath: str | Path) -> tuple[dict, dict]:
    with path(fpath, is_file=True).open(mode="r") as fd:
        json_file = json.load(fd)

        tail_templates = json_file["tail"]
        head_templates = json_file["head"]

        assert isinstance(tail_templates, dict) and isinstance(head_templates, dict)

    return tail_templates, head_templates


@dataclass
class Config:
    # data configuration
    split: Literal["validation", "test"]
    task_limit: int | None

    # model configuration
    model_path: str
    tensor_parallel_size: int

    # meta information
    system_prompt_path: str
    prompt_templates_path: str

    # TODO sample params

    # --- persistence

    def save(self, to: Path | str):
        """Speichert RunConfig mit Namen config_name im Ordner path, default: run_configurations."""
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
        """Gibt Infos zur RunConfig."""
        rep = f"""
        run configuration:
          - system prompt: {self.system_prompt_path}
          - question prompts: {self.prompt_templates_path}
          - model path: {self.model_path} (tps={self.tensor_parallel_size})
        """

        return textwrap.dedent(rep).strip()


@dataclass
class Prompt:
    task: Task
    mention: str
    relation: str
    body: str


@dataclass
class Runner:
    ds: IRT2
    model: Model

    config: Config
    out_dir: Path
    search_splits: tuple[Split, ...]

    system_prompt: str

    tail_tasks: Tasks
    tail_templates: dict

    head_tasks: Tasks
    head_templates: dict

    # --- prompt helper

    def _prompt_assemble(
        self,
        lookup: dict,
        mention: str,
        relation: str,
    ) -> str:
        if relation in lookup:
            content = lookup[relation].format(m=mention)

        else:
            self._error(f"relation {relation} not in templates - using generic")
            content = self.tail_templates["generic"].format(mention, relation)

        return f"{self.system_prompt} {content}"

    def _prompt_gen(
        self,
        tasks: Iterable,
        prompt_templates: dict,
    ) -> Generator[tuple[Prompt, set[VID]], None, None]:
        for (mid, rid), gt_vids in tasks:
            mention = self.ds.mentions[mid]
            relation = self.ds.relations[rid].split(":")[1]

            body = self._prompt_assemble(
                lookup=prompt_templates,
                mention=mention,
                relation=relation,
            )

            prompt = Prompt(
                task=(mid, rid),
                mention=mention,
                relation=relation,
                body=body,
            )

            yield prompt, gt_vids

    # --- context

    def __enter__(self):
        self._ctx_error_count = 0

        self._ctx_trace_log = (self.out_dir / "log-trace.txt").open(mode="w")
        self._ctx_error_log = (self.out_dir / "log-error.txt").open(mode="w")
        self._ctx_model_answers = gzip.open(
            filename=self.out_dir / "model-predictions.jsonl.gz",
            mode="wb",
        )

        self._trace("entered runner context")
        return self

    def __exit__(self, *_):
        ilp.console.log(f"leaving runner context; {self._ctx_error_count} total errors")

        ilp.console.log(f"closing file descriptors")
        self._ctx_trace_log.close()
        self._ctx_error_log.close()

    # --- logging

    def _write(self, fd, msg):
        # ts = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
        # fd.write(f"[{ts}] {msg}\n")
        fd.write(f"{msg}\n")

    def _trace(self, *msg: str):
        for s in msg:
            self._write(self._ctx_trace_log, s)

    def _error(self, *msg: str):
        self._ctx_error_count += 1
        for s in msg:
            self._write(self._ctx_error_log, s)

    # ---

    def _safe_parse_answer(
        self,
        prompt: Prompt,
        output: str,
    ) -> list[str]:
        try:
            return parse_answer(output)

        # parse failure: log out trace and skip task
        except Exception as exc:
            self._error(
                "-" * 80,
                f"{exc} occurred for: {prompt}",
                f"model output: {output}",
            )
            print_exc(file=self._ctx_error_log)
        return []

    def predict(
        self,
        tasks: Tasks,
        prompt_templates: dict,
    ) -> Predictions:
        # generator chain for batched processing

        prompts = self._prompt_gen(
            tasks=islice(tasks.items(), self.config.task_limit),
            prompt_templates=prompt_templates,
        )

        # ensure batched processing, however, it returns only after
        # all generation is done (TODO stream processing?)
        prompts, gt = zip(*prompts)
        outputs = self.model.prompt(prompt.body for prompt in prompts)

        preds = []
        for prompt, output, gt_vids in zip(prompts, outputs, gt):
            mentions = self._safe_parse_answer(prompt, output)
            if not mentions:
                preds.append((prompt.task, []))
                continue

            # obtain vertex predictions
            # TODO upstream; ordered result lists
            pr_vids = self.ds.find_by_mention(
                *mentions,
                splits=self.search_splits,
            )

            # assign arbitrary scores
            preds.append((prompt.task, [(vid, 1) for vid in pr_vids]))

            # --- logging

            # write answer to prompt log
            rep = {"prompt": asdict(prompt), "output": output}
            self._ctx_model_answers.write(orjson.dumps(rep) + b"\n")

            # write answer to trace log
            self._trace(
                "-" * 80,
                "\n  -".join(f"{k}: {v}" for k, v in asdict(prompt).items()),
                f"model output: {output}",
                f"parsed mentions: {', '.join(mentions)}",
                f"proposed vertices: {', '.join(self.ds.vertices[vid] for vid in pr_vids)}",
                f"true vertices: {', '.join(self.ds.vertices[vid] for vid in gt_vids)}",
                f"{len(gt_vids & pr_vids)}/{len(gt_vids)} vids are correct",
                "\n",
            )

        return preds

    def predict_all(self):
        self._trace("running predict() for tail tasks")
        tail_preds = self.predict(
            tasks=self.tail_tasks,
            prompt_templates=self.tail_templates,
        )

        self._trace("running predict() for head tasks")
        head_preds = self.predict(
            tasks=self.head_tasks,
            prompt_templates=self.head_templates,
        )


# formerly: test_run.py:_run_benchmark
def run(
    dataset: IRT2,
    config: Config,
    sampling_params: SamplingParams,
    result_folder: str | Path,
):
    """Testet Kombination aus RunConfig und Model und erstellt Evaluation"""

    out = path(result_folder, create=True)
    config.save(to=out / "run-config.yaml")

    model = model_prompter.Model(
        path=config.model_path,
        params=sampling_params,
        tensor_parallel_size=config.tensor_parallel_size,
    )

    model.load_model()

    tail_tasks, head_tasks = {
        "validation": (
            dataset.open_kgc_val_tails,
            dataset.open_kgc_val_heads,
        ),
        "test": (
            dataset.open_kgc_test_tails,
            dataset.open_kgc_test_heads,
        ),
    }[config.split]

    # TODO make splits dataset specific configuration
    assert "irt" in dataset.name.lower(), "blp has different search space"
    search_splits = (Split.train,)

    tail_templates, head_templates = load_prompt_templates(
        config.prompt_templates_path,
    )
    system_prompt = load_system_prompt(
        config.system_prompt_path,
    )

    runner = Runner(
        ds=dataset,
        model=model,
        config=config,
        out_dir=out,
        search_splits=search_splits,
        system_prompt=system_prompt,
        tail_tasks=tail_tasks,
        tail_templates=tail_templates,
        head_tasks=head_tasks,
        head_templates=head_templates,
    )

    with runner as runner:
        predictions = runner.predict_all()

    return

    # ---

    result = dict()
    result["tail_predictions"] = tail_predictions
    result["head_predictions"] = head_predictions

    evaluation = evaluate(
        ds=dataset,
        task="kgc",
        split=run_config.split,
        head_predictions=head_predictions,
        tail_predictions=tail_predictions,
    )

    print(evaluation)

    json_eval = json.dumps(evaluation)

    result_file = open(out / "result.txt", "w")
    result_file.write(json_eval)
    result_file.write("\n")

    with open(out / "result.pkl", "wb") as file:
        pickle.dump(result, file)
