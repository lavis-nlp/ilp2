"""Command line interface"""


import os
import sys

import irt2.loader
import pretty_errors
import pudb
import rich_click as click
from ktz.collections import path

import irt2_llm_prompter as ilp
from irt2_llm_prompter.run_config import RunConfig

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
    "--model-name",
    type=str,
    required=False,
    help="optional model name, otherwise directory name of --model-path",
)
@click.option(
    "--model-path",
    type=str,
    required=True,
    help="directory for vllm to load a model from",
)
@click.option(
    "--tensor-parallel-size",
    type=int,
    required=False,
    default=1,
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
def run_experiment(
    model_path: str,
    tensor_parallel_size: int,
    system_prompt: str,
    question_template: str,
    dataset_config: str,
    datasets: tuple[str],
    model_name: str | None = None,
):
    dsgen = irt2.loader.from_config_file(
        path(dataset_config, is_file=True),
        only=datasets,
    )

    config = RunConfig.from_paths(
        prompt_templates_path=path(question_template, is_file=True),
        system_prompt_path=path(system_prompt, is_file=True),
        model_path=path(model_path, is_dir=True),
        tensor_parallel_size=tensor_parallel_size,
    )

    ilp.console.print("\n", str(config), "\n")

    for name, dataset in dsgen:
        ilp.console.log(f"running experiments for {name}: {dataset}")


# ----------


def entry():
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
