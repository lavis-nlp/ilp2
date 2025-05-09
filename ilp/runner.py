import csv
import gzip
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from itertools import islice
from pathlib import Path
from traceback import print_exc
from typing import Callable, Generator, Iterable, Literal, Sequence

import orjson
import yaml
from irt2.dataset import IRT2
from irt2.evaluation import Predictions, Scores, evaluate
from irt2.types import MID, RID, VID, Split, Task
from ktz.collections import dflat, path
from rich.progress import track

import ilp
from ilp.config import Config
from ilp.model import ModelBase
from ilp.preprocessor import remove_stopwords, stem
from ilp.prompts import Assembler

Tasks = dict[tuple[MID, RID], set[VID]]


@dataclass
class PromptContext:
    task: Task
    mention: str
    relation: str
    prompt: str


@dataclass
class Runner:
    ds: IRT2
    model: ModelBase
    assembler: Assembler
    transformations: list[Callable[[str], str]]

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
        mid2str_original: dict[MID, str],
    ) -> tuple[list[str], list[str]]:
        vid2mids: dict[VID, set[MID]] = {}

        if self.ds.name.startswith("IRT2"):
            vid2mids |= self.ds.idmap.vid2mids[Split.train]

        if self.ds.name.startswith("BLP"):
            splits = (Split.train, Split.valid)
            if self.config.dataset_split == "test":
                splits += (Split.test,)

            for split in splits:
                vid2mids |= self.ds.idmap.vid2mids[split]

        return (
            [
                self.ds.idmap.mid2str[mid]
                for vid in gt_vids
                for mid in vid2mids[vid]
                if mid in self.ds.idmap.mid2str
            ],
            [mid2str_original[mid] for vid in gt_vids for mid in vid2mids[vid]],
        )

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

            if self.config.give_true_candidates:
                gt_mentions_transformed, _ = self._create_true_answer(
                    gt_vids,
                    self.ds.idmap.mid2str,
                )

                candidates = ", ".join(gt_mentions_transformed)

                prompt = self.assembler.assemble(
                    direction=direction,
                    mid=mid,
                    mention=mention,
                    rid=rid,
                    relation=relation,
                    candidates=candidates,
                )
            else:
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

    def transform(self, s: str) -> str:
        if s.isdigit():
            return s
        for fn in self.transformations:
            s = fn(s)
        return s

    # ---  PREDICTIONS

    # all _predict* functions return:
    # pr_vids: Sequence[Sequence[VID]] containing the vids proposed by the model
    #   - scores are assigned automatically descending on order of the sequence
    #   - subsequences get the same score
    #   - e.g. [[A,B], [C, ]] -> [(A, 2), (B, 2), (C, 1)]

    # --

    # mode: default
    #   - ask model, obtain mentions, ground mentions by search
    #   - mentions found by name are ranked higher than all other
    #   - order is retained for both named and mentioned vids
    def _parse_default(self, raw_pr_mentions: list):
        return list(map(self.transform, raw_pr_mentions))

    def _predict_default(self, pr_mentions: list) -> Sequence:
        # retain order of proposals
        known, named = set(), []
        for mention in pr_mentions:
            if mention not in self.name2vid:
                continue
            vid = self.name2vid[mention]
            if vid in known:
                continue

            known.add(vid)
            named.append([vid])

        mentioned = []
        for mention in pr_mentions:
            found: set[VID] = self.ds.find_by_mention(
                mention,
                splits=self.search_splits,
            )

            found -= known
            mentioned.append(list(found))
            known |= found

        return named + mentioned

    # mode: prompt re-ranking
    #  - model gets N pre-ranked entities
    #  - model is only allowed to select from list
    #  - new order is determined by model output
    #  - proposals ignored by the model are appended
    #  - other >N pre-ranking results are appended after
    def _parse_prr(self, raw_pr_mentions: list):
        return list(filter(str.isdigit, raw_pr_mentions))

    def _predict_prr(
        self,
        pr_mentions: list[str],
        direction: Literal["head", "tail"],
        task: Task,
    ) -> Sequence:
        preranked_vids = self.assembler.get_ranked_vids(
            direction=direction,
            task=task,
        )

        known, vids = set(), []
        for idx in map(int, pr_mentions):
            # only allow proposed
            if idx > self.config.n_candidates:
                continue

            # assuming they are proposed with index 0
            vid = preranked_vids[idx]

            # skip duplicates
            if vid in known:
                continue

            vids.append([vid])
            known.add(vid)

        vids += list([vid] for vid in preranked_vids if vid not in known)
        return vids

    # mode: full re-ranking
    def _parse_frr(self, raw_pr_mentions: list):
        return list(map(self.transform, raw_pr_mentions))

    def _predict_frr(
        self,
        pr_mentions: list[str],
        direction: Literal["head", "tail"],
        task: Task,
    ):
        preranked_vids = self.assembler.get_ranked_vids(
            direction=direction,
            task=task,
        )

        known, vids = set(), []
        for pred in pr_mentions:
            # (1) model referenced pre-ranker list
            if pred.isdigit():
                idx = int(pred)
                if idx > self.config.n_candidates:
                    continue

                vid = preranked_vids[idx]
                if vid in known:
                    continue

                vids.append([vid])
                known.add(vid)
                continue

            # (2) try to look up by name
            vid = self.name2vid.get(pred)
            if vid in known:
                continue

            if vid is not None:
                vids.append([vid])
                known.add(vid)
                continue

            # (3) look up by mention
            matched_vids = self.ds.find_by_mention(
                pred,
                splits=self.search_splits,
            )
            matched_vids -= known

            vids.append(list(matched_vids))
            known |= matched_vids

        # append remaining pre-ranked candidates
        vids += list([vid] for vid in preranked_vids if vid not in known)
        return vids

    # mode: ranker-results
    def _predict_rr(self, ctx, direction) -> Scores:
        preranked_vids = self.assembler.get_ranked_vids(
            direction=direction,
            task=ctx.task,
        )
        return [(vid, score) for score, vid in enumerate(preranked_vids[::-1])]

    # modes in which model answers are parsed
    def _predict(self, pr_mentions, direction, task):
        match self.config.mode:
            case "default":
                return self._predict_default(
                    pr_mentions,
                )

            case "prompt-re-ranking":
                return self._predict_prr(
                    pr_mentions,
                    direction=direction,
                    task=task,
                )

            case "full-re-ranking":
                return self._predict_frr(
                    pr_mentions,
                    direction=direction,
                    task=task,
                )

        assert False

    def _parse(self, ctx, output) -> list:
        self._ctx_stats["parse_attempts"] += 1
        raw_pr_mentions = self._safe_parse_answer(ctx, output)

        if len(raw_pr_mentions) == 0:
            self._ctx_stats["parse_errors"] += 1

        match self.config.mode:
            case "default":
                return self._parse_default(raw_pr_mentions)

            case "prompt-re-ranking":
                return self._parse_prr(raw_pr_mentions)

            case "full-re-ranking":
                return self._parse_frr(raw_pr_mentions)

        assert False

    def _prompt(self, prompts):
        lis = list(prompts)
        ilp.console.log(f"querying model with {len(lis)} prompts...")

        breakpoint()

        ts_start = datetime.now()
        res = list(self.model.prompt(self.config, lis))
        ts_end = datetime.now()

        t_delta = ts_end - ts_start
        t_per_prompt = t_delta / len(lis)

        ilp.console.log(f"finished, took {t_delta} ({t_per_prompt} per prompt)")
        return res

    def predict(
        self,
        tasks: Tasks,
        direction: Literal["head", "tail"],
    ) -> Predictions:
        ilp.console.log(f"predicting {direction}s")

        # generator chain for batched processing
        prompt_gen = self._prompt_gen(
            direction=direction,
            tasks=islice(
                tasks.items(),
                self.config.dataset_limit_tasks,
            ),
        )

        # ensure batched processing
        ctxs, gt = zip(*prompt_gen)

        if not self.dry_run and self.config.mode != "ranker-results":
            if not self.re_evaluate:
                # load model before tracking starts
                self.model.load()
                outputs = self._prompt(ctx.prompt for ctx in ctxs)
            else:
                outputs = self._load_model_outputs()
        else:
            outputs = self._create_empty_outputs(ctxs)

        # --

        # save state
        mid2str_original = self.ds.idmap.mid2str.copy()

        # overwrite with transformed mentions
        self.ds.idmap.mid2str = {}
        for mid, s in mid2str_original.items():
            transformed = self.transform(s)
            if transformed != "":
                self.ds.idmap.mid2str[mid] = transformed

        # invalidate cache
        try:
            del self.ds.idmap.str2mids
        except AttributeError:
            ...

        preds: Predictions = []

        zipped = zip(ctxs, outputs, gt)
        tracked = track(
            zipped,
            description=f'{"parsing":12s}',
            total=len(ctxs),
            console=ilp.console,
            disable=ilp.debug,
        )

        for ctx, output, gt_vids in tracked:

            if not self.re_evaluate:
                rep = {"ctx": asdict(ctx), "output": output}
                self._ctx_model_answers.write(orjson.dumps(rep) + b"\n")  # type:ignore

            # no LLM involved, just use pre-ranking
            if self.config.mode == "ranker-results":
                preds.append((ctx.task, self._predict_rr(ctx, direction)))
                continue

            # LLM involved, parse model answer

            gt_mentions_transformed, gt_mentions = self._create_true_answer(
                gt_vids,
                mid2str_original,
            )

            # dry run: oracle
            pr_mentions: list[str]

            if self.dry_run and self.config.mode == 'default':
                pr_mentions = gt_mentions_transformed
            if self.dry_run:
                pr_mentions = list(map(str, range(10)))
            else:
                pr_mentions = self._parse(ctx, output)

            # if ctx.task == (13408, 2):
            #     breakpoint()

            # obtain vertex predictions
            pr_vids: Sequence[Sequence[VID]]
            pr_vids = self._predict(
                pr_mentions,
                direction,
                ctx.task,
            )

            # assign scores (removes empty lists)
            scored = enumerate(filter(len, pr_vids[::-1]))
            scored = [(vid, score) for score, vids in scored for vid in vids]
            preds.append((ctx.task, scored))

            k = 10
            topk = sorted(scored, key=lambda t: t[1], reverse=True)[:k]
            scored_fmt = [f"{score}:{self.ds.vertices[vid]}" for vid, score in topk]
            topk_vids = {vid for vid, _ in topk}

            self._trace(
                "-" * 80,
                "\n  -".join(f"{k}: {v}" for k, v in asdict(ctx).items()),
                f"model output: {output}",
                f"transformed parsed mentions: {', '.join(pr_mentions)}",
                # f"additional proposed vertices: {', '.join(additionally_proposed_names)}",  TODO
                f"true mentions: {', '.join(gt_mentions)}",
                f"transformed true mentions: {', '.join(gt_mentions_transformed)}",
                f"scored vertices (top-{k}): {', '.join(scored_fmt)}",
                f"true vertices (top-{k}): {', '.join(self.ds.vertices[vid] for vid in gt_vids)}",
                f"{len(gt_vids & topk_vids)}/{len(gt_vids)} vids are correct",
                f"{len(set(topk_vids) - set(gt_vids))} are incorrectly predicted vertices",
                "\n",
            )

        # restore state
        self.ds.idmap.mid2str = mid2str_original

        return preds

    def predict_all(self) -> dict:

        # TODO maybe use post_init
        self.name2vid: dict[str, VID] = {
            self.transform(self.ds.idmap.vid2str[vid].split(":")[1]): vid
            for vid in self.ds.closed_vertices
        }

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
    model: ModelBase,
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
    }[config.dataset_split]
    assert len(tasks) == 2 and "head" in tasks and "tail" in tasks

    # TODO make splits dataset specific configuration
    if dataset.name.startswith("IRT"):
        search_splits = (Split.train,)
    elif dataset.name.startswith("BLP"):
        search_splits = (Split.train, Split.valid)
        if config.dataset_split == "test":
            search_splits += (Split.test,)
    else:
        assert False

    transformations: list[Callable[[str], str]] = []

    transformations += [str.lower]
    transformations += [str.strip]

    def remove_dots(str):
        return str.replace(".", "")

    transformations += [remove_dots]

    if config.stopwords_path != None:
        with open(config.stopwords_path, "r", encoding="utf-8") as stopword_file:
            transformations += [
                partial(
                    remove_stopwords,
                    stopwords=stopword_file.read().split(","),
                )
            ]

    if config.use_stemmer:
        transformations += [stem]

    assembler = Assembler.from_paths(
        dataset=dataset,
        mode=config.mode,
        split_str=config.dataset_split,
        search_splits=search_splits,
        dataset_name=dataset.name,
        template_path=config.prompt_template,
        system_path=config.prompt_system,
        question_path=config.prompt_question,
        texts_head_path=config.dataset_texts_head,
        texts_tail_path=config.dataset_texts_tail,
        n_candidates=config.n_candidates,
        mentions_per_candidate=config.mentions_per_candidate,
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
        transformations=transformations,
    )

    ilp.console.log("create predictions and evaluate")
    with runner as runner:
        predictions = runner.predict_all()

    report = evaluate(
        ds=dataset,
        task="kgc",
        split=config.dataset_split,
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


def wrapped_run(
    mode: str,
    dataset: IRT2,
    model: ModelBase,
    config: Config,
    dry_run: bool = False,
    output_prefix: str = "",
):
    ts_start_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    dirparent = Path(model.path).name + ("-dry" if dry_run else "")
    dsname = dataset.name.replace("/", "_").lower()
    dirname = f"{output_prefix}{mode}-{dsname}-{ts_start_str}"

    out = path(
        path("data") / "experiments" / dirparent / dirname,
        # create=True,
    )

    ilp.console.log(f"write results to {out}")

    try:
        run(
            model=model,
            dataset=dataset,
            config=config,
            result_folder=out,
            dry_run=dry_run,
        )
    except Exception as exc:
        ilp.console.log(f"{exc} occurred! writing postmortem")
        with (out / "postmortem.txt").open(mode="w") as fd:
            print_exc(file=fd)

        if ilp.debug:
            raise
