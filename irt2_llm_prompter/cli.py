"""Command line interface"""


import os
import sys
from datetime import datetime
from traceback import print_exc
from typing import Literal

import irt2.loader
import pretty_errors
import pudb
import rich_click as click
from ktz.collections import path

import irt2_llm_prompter as ilp
from irt2_llm_prompter.model import Model
from irt2_llm_prompter.runner import Config, run

os.environ["PYTHONBREAKPOINT"] = "pudb.set_trace"


pretty_errors.configure(
    filename_display=pretty_errors.FILENAME_EXTENDED,
    lines_after=2,
    line_number_first=True,
)


@click.group()
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    default=False,
    help="suppress console output",
)
@click.option(
    "-d",
    "--debug",
    is_flag=True,
    default=False,
    help="activate debug mode (drop into pudb on error)",
)
def main(quiet: bool, debug: bool):
    """Use irt2m from the command line."""
    ilp.debug = debug
    ilp.console.quiet = quiet

    ilp.console.log(f"IRT2 LLM Prompting ({ilp.version})")
    ilp.console.log(f"executing from: {os.getcwd()}")


# formerly /run_test.py


@main.command(name="run-experiment")
@click.option(
    "--split",
    type=click.Choice(["validation", "test"]),
    required=True,
    help="run on test or validation",
)
@click.option(
    "--model",
    type=str,
    required=False,
    help="directory for vllm to load a model from",
)
@click.option(
    "--tensor-parallel-size",
    type=int,
    required=False,
    default=1,
)
@click.option(
    "--prompt-template",
    type=str,
    required=True,
    help="prompt template - see conf/prompts/template",
)
@click.option(
    "--system-prompt",
    type=str,
    required=True,
    help="system prompt - see conf/prompts/system",
)
@click.option(
    "--question-template",
    type=str,
    required=True,
    help="question template - see conf/prompts/question",
)
@click.option(
    "--dataset-config",
    type=str,
    help="some lib/irt2/conf/datasets yaml file",
)
@click.option(
    "--datasets",
    type=str,
    multiple=True,
    help="select keys from dataset-config",
)
@click.option(
    "--texts-head",
    type=str,
    help="optional, supplemental text, make sure to select a single dataset",
)
@click.option(
    "--texts-tail",
    type=str,
    help="optional, supplemental text, make sure to select a single dataset",
)
@click.option(
    "--parser",
    nargs=1,
    default="json",
    required=False,
    type=click.Choice(["json","csv"], case_sensitive=False),
    help="optinal, choose parser corresponding to prompts",
)
@click.option(
    "--limit-tasks",
    type=int,
    required=False,
    help="run at most n samples per direction (head/tail)",
)
@click.option(
    "--output-prefix",
    type=str,
    required=False,
    default="",
    help="prefix put before the timestamp of the result directory",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="do not load the model but instead always answer right",
)
@click.option("--sampling-temperature", type=float)
@click.option("--sampling-top-p", type=float)
@click.option("--sampling-use-beam-search", type=bool)
@click.option("--sampling-early-stopping", type=bool)
@click.option("--sampling-best-of", type=int)
@click.option("--sampling-max-tokens", type=int)
@click.option("--sampling-repetition-penalty", type=float)
def run_experiment(
    split: Literal["validation", "test"],
    model: str,
    tensor_parallel_size: int,
    prompt_template: str,
    system_prompt: str,
    question_template: str,
    dataset_config: str,
    datasets: tuple[str],
    texts_head: str | None,
    texts_tail: str | None,
    parser: Literal["json", "csv"],
    limit_tasks: int | None,
    output_prefix: str,
    dry_run: bool = False,
    **sampling_params,
):
    config = Config(
        # dataset related
        dataset_path=dataset_config,
        split=split,
        task_limit=limit_tasks,
        dataset_texts_head=texts_head,
        dataset_texts_tail=texts_tail,
        # model related
        model_path=model,
        tensor_parallel_size=tensor_parallel_size,
        parser=parser,
        # prompt related
        prompt_template_path=prompt_template,
        prompt_system_path=system_prompt,
        prompt_question_path=question_template,
        # sampling params
        **{
            k.replace("sampling_", ""): v
            for k, v in sampling_params.items()
            if v is not None
        },
    )

    ilp.console.print("\n", str(config), "\n")

    dsgen = irt2.loader.from_config_file(
        path(dataset_config, is_file=True),
        only=datasets,
    )

    model_instance = Model.from_config(config)
    for name, dataset in dsgen:
        ilp.console.log(f"running experiments for {name}: {dataset}")
        ts_start_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        dirparent = path(model).name + ("-dry" if dry_run else "")
        dirname = f"{output_prefix}{ts_start_str}"

        out = path(
            path("data") / "experiments" / dirparent / dirname,
            create=True,
        )

        ilp.console.log(f"write results to {out}")

        try:
            run(
                model=model_instance,
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


@main.command(name="re-evaluate")
@click.argument(
    "folder",
    nargs=-1,
)
def main_reevaluate(folder: str):
    for fpath in (path(f, is_dir=True) for f in folder):
        ilp.console.log(f"re-evaluating {fpath}")

        config = Config.load(fpath / "run-config.yaml")
        ilp.console.print("\n", str(config), "\n")

        # TODO


# ----------


def entry():
    # speak friend and enter

    try:
        main()

    except Exception as exc:
        if not ilp.debug:
            raise exc

        ilp.console.log("debug: catched exception, starting debugger")
        ilp.console.log(str(exc))

        _, _, tb = sys.exc_info()
        pudb.post_mortem(tb)

        raise exc
