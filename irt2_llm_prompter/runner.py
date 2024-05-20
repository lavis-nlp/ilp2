import csv
import gzip
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import islice
from pathlib import Path
from traceback import print_exc
from typing import Generator, Iterable, Literal

import orjson
import yaml
from irt2.dataset import IRT2
from irt2.evaluation import Predictions, evaluate
from irt2.types import MID, RID, VID, Split, Task
from ktz.collections import dflat, path

import irt2_llm_prompter as ilp
from irt2_llm_prompter.model_prompter import Model

Tasks = dict[tuple[MID, RID], set[VID]]


# TODO prompt generation and vertex lookup
# should become the task of the model, not the runner


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


def load_prompt_templates(ds: IRT2, fpath: str | Path):
    with path(fpath, is_file=True).open(mode="r") as fd:
        prompts = yaml.safe_load(fd)

    for conf in prompts:
        for name in conf["datasets"]:
            if name != ds.name:
                continue

            return (
                conf["prompts"]["tail"],
                conf["prompts"]["head"],
            )

    assert False, f"{ds.name} not found in template"


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
    dataset_path: str

    # sampling params (beam search)
    use_beam_search: bool = True
    early_stopping: bool = False  # must be False for random sampling
    best_of: int = 2  # must be 1 for greedy sampling (t=0)

    # sampling params (random sampling)
    temperature: float = 0  # greedy if beam_search is False and set to 0
    top_p: float = 1  # consider tokens until their cum. prob. reaches

    max_tokens: int = 512

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
        self._ctx_model_answers.close()

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
        dry_run: bool = False,
    ) -> Predictions:
        # generator chain for batched processing

        prompts = self._prompt_gen(
            tasks=islice(tasks.items(), self.config.task_limit),
            prompt_templates=prompt_templates,
        )

        # ensure batched processing, however, it returns only after
        # all generation is done (TODO stream processing?)
        prompts, gt = zip(*prompts)

        if not dry_run:
            outputs = self.model.prompt(prompt.body for prompt in prompts)
        else:
            outputs = ["" for _ in range(len(prompts))]

        preds = []
        for prompt, output, gt_vids in zip(prompts, outputs, gt):
            if not dry_run:
                mentions = self._safe_parse_answer(prompt, output)

            else:
                mentions = [
                    self.ds.idmap.mid2str[mid]
                    for vid in gt_vids
                    for mid in self.ds.idmap.vid2mids[vid]
                ]

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

            rep = {"prompt": asdict(prompt), "output": output}
            self._ctx_model_answers.write(orjson.dumps(rep) + b"\n")
            self._trace(
                "-" * 80,
                "\n  -".join(f"{k}: {v}" for k, v in asdict(prompt).items()),
                f"model output: {output}",
                f"parsed mentions: {', '.join(mentions)}",
                f"proposed vertices: {', '.join(self.ds.vertices[vid] for vid in pr_vids)}",
                f"true vertices: {', '.join(self.ds.vertices[vid] for vid in gt_vids)}",
                f"{len(gt_vids & pr_vids)}/{len(gt_vids)} vids are correct",
                f"{len(pr_vids - gt_vids)} are incorrectly predicted vertices",
                "\n",
            )

        return preds

    def predict_all(
        self,
        dry_run: bool = False,
    ) -> dict[Literal["head", "tail"], Predictions]:
        self._trace("running predict() for tail tasks")
        tail_preds = self.predict(
            tasks=self.tail_tasks,
            prompt_templates=self.tail_templates,
            dry_run=dry_run,
        )

        self._trace("running predict() for head tasks")
        head_preds = self.predict(
            tasks=self.head_tasks,
            prompt_templates=self.head_templates,
            dry_run=dry_run,
        )

        return dict(
            tail=tail_preds,
            head=head_preds,
        )


# formerly: test_run.py:_run_benchmark
def run(
    model: Model,
    dataset: IRT2,
    config: Config,
    result_folder: str | Path,
    dry_run: bool = False,
) -> dict:
    ts_start = datetime.now()

    out = path(result_folder, create=True)
    config.save(to=out / "run-config.yaml")

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
        dataset,
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

    ilp.console.log("create predictions and evaluate")
    with runner as runner:
        predictions = runner.predict_all(dry_run)

    report = evaluate(
        ds=dataset,
        task="kgc",
        split=config.split,
        head_predictions=predictions["head"],
        tail_predictions=predictions["tail"],
    )

    ts_end = datetime.now()

    # --- logging

    ilp.console.log(f"finished evaluation, run took {ts_end - ts_start}")
    ilp.console.print(report["all"]["micro"])

    with (out / "evaluation-report.yaml").open(mode="w") as fd:
        yaml.safe_dump(report, fd)

    with (out / "evaluation-report.csv").open(mode="w") as fd:
        csv_meta = (
            ("dataset", dataset.name),
            ("run", out.name),
            ("duration", ts_end - ts_start),
            ("start", ts_start),
            ("end", ts_end),
        )

        csv_meta += tuple(asdict(config).items())
        csv_report = tuple(sorted(dflat(report, sep=" ").items()))

        writer = csv.writer(fd)
        for row in zip(*(csv_meta + csv_report)):
            writer.writerow(row)

    return report
