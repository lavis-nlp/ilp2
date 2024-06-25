import csv
import gzip
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
from irt2_llm_prompter.model import Model
from irt2_llm_prompter.prompts import Assembler

Tasks = dict[tuple[MID, RID], set[VID]]


@dataclass
class Config:
    # data configuration
    dataset_path: str
    split: Literal["validation", "test"]
    task_limit: int | None
    dataset_texts_head: str | None
    dataset_texts_tail: str | None

    # model configuration
    model_path: str
    tensor_parallel_size: int
    parser: Literal["json", "csv"]

    # prompt templates
    prompt_template_path: str  # conf/prompts/template
    prompt_system_path: str  # conf/prompts/system
    prompt_question_path: str  # conf/prompts/question

    # sampling params (beam search)
    use_beam_search: bool = True
    early_stopping: bool = False  # must be False for random sampling
    best_of: int = 2  # must be 1 for greedy sampling (t=0)

    # sampling params (random sampling)
    temperature: float = 0  # greedy if beam_search is False and set to 0
    top_p: float = 1.0  # consider tokens until their cum. prob. reaches
    repetition_penalty: float = 1.0  # penalize new tokens if they appeared before

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
class PromptContext:
    task: Task
    mention: str
    relation: str
    prompt: str


@dataclass
class Runner:
    ds: IRT2
    model: Model
    assembler: Assembler

    config: Config
    out_dir: Path
    search_splits: tuple[Split, ...]

    tasks: dict[Literal["head", "tail"], Tasks]

    dry_run: bool
    re_evaluate: bool

    # --- context

    def __enter__(self):
        self._ctx_stats = {
            "error_count": 0,
            "parse_attempts": 0,
            "parse_errors": 0,
        }

        self._ctx_trace_log = (self.out_dir / "log-trace.txt").open(mode="w")
        self._ctx_error_log = (self.out_dir / "log-error.txt").open(mode="w")
        self._ctx_model_answers = gzip.open(
            filename=self.out_dir / "model-predictions.jsonl.gz",
            mode=("rb" if self.re_evaluate else "wb"),
        )

        self._trace("entered runner context")
        return self

    def __exit__(self, *_):
        error_count = self._ctx_stats["error_count"]
        ilp.console.log(f"leaving runner context; {error_count} total errors")

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
        self._ctx_stats["error_count"] += 1
        for s in msg:
            self._write(self._ctx_error_log, s)

    # ---

    def _create_true_answer(
        self,
        gt_vids: set[VID],
    ) -> list[str]:
        vid2mids: dict[VID, set[MID]] = {}

        if self.ds.name.startswith("IRT2"):
            vid2mids |= self.ds.idmap.vid2mids[Split.train]

        if self.ds.name.startswith("BLP"):
            splits = (Split.train, Split.valid)
            if self.config.split == "test":
                splits += (Split.test,)

            for split in splits:
                vid2mids |= self.ds.idmap.vid2mids[split]

        return [self.ds.idmap.mid2str[mid] for vid in gt_vids for mid in vid2mids[vid]]

    def _safe_parse_answer(
        self,
        prompt: PromptContext,
        output: str,
    ) -> list[str]:
        try:
            return self.model.parse(output)

        # parse failure: log out trace and skip task
        except Exception as exc:
            self._error(
                "-" * 80,
                f"{exc} occurred for: {prompt}",
                f"model output: {output}",
            )
            print_exc(file=self._ctx_error_log)
        return []

    def _load_model_outputs(self) -> list[str]:
        assert False, "untested"
        return [orjson.loads(rep)["output"] for rep in self._ctx_model_answers]

    def _create_empty_outputs(self, ctxs: Iterable[PromptContext]) -> list[str]:
        # realise prompts here at the latest even if they behave
        # lazily - this is done to assert that the templating works
        outputs = [ctx.prompt for ctx in ctxs]
        outputs = ["" for _ in outputs]
        return outputs

    def _prompt_gen(
        self,
        direction: Literal["head", "tail"],
        tasks: Iterable,
    ) -> Generator[tuple[PromptContext, set[VID]], None, None]:
        for (mid, rid), gt_vids in tasks:
            mention = self.ds.mentions[mid]
            relation = self.ds.relations[rid]

            # self._error(f"relation {relation} not in templates - using generic")
            # content = self.tail_templates["generic"].format(mention, relation)

            prompt = self.assembler.assemble(
                direction=direction,
                mid=mid,
                mention=mention,
                rid=rid,
                relation=relation,
            )

            ctx = PromptContext(
                task=(mid, rid),
                mention=mention,
                relation=relation,
                prompt=prompt,
            )

            yield ctx, gt_vids

    def predict(
        self,
        tasks: Tasks,
        direction: Literal["head", "tail"],
    ) -> Predictions:
        # generator chain for batched processing

        prompt_gen = self._prompt_gen(
            direction=direction,
            tasks=islice(tasks.items(), self.config.task_limit),
        )

        # ensure batched processing, however, it returns only after
        # all generation is done (TODO stream processing?)
        ctxs, gt = zip(*prompt_gen)

        # TODO messy, hard to understand structure
        if not self.dry_run:
            if not self.re_evaluate:
                outputs = self.model.prompt(ctx.prompt for ctx in ctxs)
            else:
                outputs = self._load_model_outputs()
        else:
            outputs = self._create_empty_outputs(ctxs)

        # --

        preds = []
        for ctx, output, gt_vids in zip(ctxs, outputs, gt):
            gt_mentions = self._create_true_answer(gt_vids)

            if not self.re_evaluate:
                rep = {"ctx": asdict(ctx), "output": output}
                self._ctx_model_answers.write(orjson.dumps(rep) + b"\n")  # type:ignore

            if not self.dry_run:
                self._ctx_stats["parse_attempts"] += 1
                pr_mentions = self._safe_parse_answer(ctx, output)
                if len(pr_mentions) == 0:
                    self._ctx_stats["parse_errors"] += 1

            else:
                pr_mentions = gt_mentions

            if not pr_mentions:
                preds.append((ctx.task, []))
                continue

            # obtain vertex predictions
            # TODO upstream; ordered result lists
            pr_vids = self.ds.find_by_mention(
                *pr_mentions,
                splits=self.search_splits,
            )

            # assign arbitrary scores
            preds.append((ctx.task, [(vid, 1) for vid in pr_vids]))

            # --- logging

            self._trace(
                "-" * 80,
                "\n  -".join(f"{k}: {v}" for k, v in asdict(ctx).items()),
                f"model output: {output}",
                f"parsed mentions: {', '.join(pr_mentions)}",
                f"true mentions: {', '.join(gt_mentions)}",
                f"proposed vertices: {', '.join(self.ds.vertices[vid] for vid in pr_vids)}",
                f"true vertices: {', '.join(self.ds.vertices[vid] for vid in gt_vids)}",
                f"{len(gt_vids & pr_vids)}/{len(gt_vids)} vids are correct",
                f"{len(pr_vids - gt_vids)} are incorrectly predicted vertices",
                "\n",
            )

        return preds

    def predict_all(self) -> dict:
        result = {}
        for direction in ("head", "tail"):
            self._trace(f"running predict() for {direction} tasks")
            result[direction] = self.predict(
                tasks=self.tasks[direction],
                direction=direction,
            )

        result |= {"stats": dict(self._ctx_stats)}

        # ---

        self._trace(result["stats"])
        ilp.console.log("result stats:", result["stats"])

        return result


def run(
    model: Model,
    dataset: IRT2,
    config: Config,
    result_folder: str | Path,
    dry_run: bool = False,
    re_evaluate: bool = False,
) -> dict:
    ts_start = datetime.now()

    out = path(result_folder, create=True)
    config.save(to=out / "run-config.yaml")

    tasks: dict[str, Tasks] = {
        "validation": dict(
            tail=dataset.open_kgc_val_tails,
            head=dataset.open_kgc_val_heads,
        ),
        "test": dict(
            tail=dataset.open_kgc_test_tails,
            head=dataset.open_kgc_test_heads,
        ),
    }[config.split]
    assert len(tasks) == 2 and "head" in tasks and "tail" in tasks

    # TODO make splits dataset specific configuration
    if dataset.name.startswith("IRT"):
        search_splits = (Split.train,)
    elif dataset.name.startswith("BLP"):
        search_splits = (Split.train, Split.valid)
        if config.split == "test":
            search_splits += (Split.test,)
    else:
        assert False

    assembler = Assembler.from_paths(
        dataset_name=dataset.name,
        template_path=config.prompt_template_path,
        system_path=config.prompt_system_path,
        question_path=config.prompt_question_path,
        texts_head_path=config.dataset_texts_head,
        texts_tail_path=config.dataset_texts_tail,
    )

    runner = Runner(
        ds=dataset,
        model=model,
        assembler=assembler,
        config=config,
        out_dir=out,
        search_splits=search_splits,
        tasks=tasks,  # type: ignore
        dry_run=dry_run,
        re_evaluate=re_evaluate,
    )

    ilp.console.log("create predictions and evaluate")
    with runner as runner:
        predictions = runner.predict_all()

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
        csv_meta += tuple(predictions["stats"].items())
        csv_report = tuple(sorted(dflat(report, sep=" ").items()))

        writer = csv.writer(fd)
        for row in zip(*(csv_meta + csv_report)):
            writer.writerow(row)

    return report
