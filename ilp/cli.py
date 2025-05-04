"""Command line interface"""

import os
import sys
from itertools import product
from typing import Literal

import irt2.loader
import pretty_errors
import pudb
import rich_click as click
from ktz.collections import path

import ilp
from ilp.model import ModelBase, VLLMModel
from ilp.runner import Config, wrapped_run

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
    "--mode",
    type=click.Choice(
        ["default", "prompt-re-ranking", "full-re-ranking", "ranker-results"],
        case_sensitive=False,
    ),
    required=True,
    help="mode in which outputs are processed, use mode specific prompts",
)
@click.option(
    "--dataset-config",
    type=str,
    help="some lib/irt2/conf/datasets yaml file",
)
@click.option(
    "--dataset-key",
    type=str,
    multiple=True,
    help="select keys from dataset-config",
)
@click.option(
    "--dataset-split",
    type=click.Choice(["validation", "test"]),
    required=True,
    help="run on test or validation",
)
@click.option(
    "--dataset-texts-head",
    type=str,
    help="optional, supplemental text, make sure to select a single dataset",
)
@click.option(
    "--dataset-texts-tail",
    type=str,
    help="optional, supplemental text, make sure to select a single dataset",
)
@click.option(
    "--dataset-limit-tasks",
    type=int,
    required=False,
    help="run at most n samples per direction (head/tail)",
)
@click.option(
    "--prompt-template",
    type=str,
    required=True,
    help="prompt template - see conf/prompts/template",
)
@click.option(
    "--prompt-system",
    type=str,
    required=True,
    help="system prompt - see conf/prompts/system",
)
@click.option(
    "--prompt-question",
    type=str,
    required=True,
    help="question template - see conf/prompts/question",
)
@click.option(
    "--model-path",
    type=str,
    required=False,
    help="directory for vllm to load a model from",
)
@click.option(
    "--model-tensor-parallel-size",
    type=int,
    required=False,
    default=1,
)
@click.option(
    "--model-gpu-memory-utilization",
    type=float,
    required=False,
    default=1.0,
)
@click.option(
    "--model-parser",
    nargs=1,
    default="csv",
    required=False,
    type=click.Choice(["json", "csv"], case_sensitive=False),
    help="optinal, choose parser corresponding to prompts",
)
@click.option(
    "--model-engine",
    default="vllm",
    required=False,
    type=click.Choice(["huggingface", "vllm"], case_sensitive=False),
    help="optional, choose the inference engine to use: 'huggingface' or 'vllm'.",
)
@click.option(
    "--model-quantization",
    type=click.Choice(
        ["fp8"],
        case_sensitive=False,
    ),
    default=None,
    show_default=True,
    help="optional, specify quantization technique for model weights",
)
@click.option(
    "--model-max-tokens",
    type=int,
    default=1024,
)
@click.option(
    "--model-use-beam-search",
    type=bool,
    default=False,
)
@click.option(
    "--stopwords-path",
    type=str,
    required=False,
    help="optional, stopword-list - see conf/stopwords",
)
@click.option(
    "--use-stemmer",
    is_flag=True,
    help="optional, use pystemmer to stem mentions",
)
@click.option(
    "--n-candidates",
    type=int,
    required=False,
    default=(0,),
    multiple=True,
    help="optional, gives the model the top n candidates",
)
@click.option(
    "--mentions-per-candidate",
    type=int,
    required=False,
    default=1,
    help="optional, amount of mentions per candidate proposed",
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
@click.option(
    "--give-true-candidates",
    is_flag=True,
    default=False,
    help="use ground truth mentions as candidates",
)
@click.option(
    "--sampling-temperature",
    type=float,
    multiple=True,
    default=(0.9,),
)
@click.option(
    "--sampling-top-p",
    type=float,
    multiple=True,
    default=(0.6,),
)
@click.option(
    "--sampling-beam-width",
    type=int,
    default=2,
)
@click.option(
    "--sampling-early-stopping",
    type=bool,
    default=True,
)
@click.option(
    "--sampling-repetition-penalty",
    type=float,
    default=1.0,
)
@click.option(
    "--sampling-length-penalty",
    type=float,
    default=1.0,
)
def run_experiment(
    mode: Literal[
        "default",
        "prompt-re-ranking",
        "full-re-ranking",
        "ranker-results",
    ],
    dataset_config: str,
    dataset_key: tuple[str],
    model_engine: Literal["vllm", "huggingface"],
    n_candidates: tuple[int],
    sampling_temperature: tuple[float],
    sampling_top_p: tuple[float],
    output_prefix: str,
    dry_run: bool = False,
    **kwargs,
):
    # iterate all combinations of sweeping parameters
    # and create a config for each

    datasets = dict(
        irt2.loader.from_config_file(
            path(dataset_config, is_file=True),
            only=dataset_key,
        )
    )

    assert model_engine == "vllm", "huggingface is disabled for now"
    model_instance = VLLMModel(
        **{
            k[len("model_") :]: v
            for k, v in kwargs.items()
            if k.startswith("model_")
            #
        }
    )

    defer_loading = (
        ilp.debug,
        dry_run,
        mode == 'ranker-results',
    )

    if not any(defer_loading):
        model_instance.load()

    sweeping = {
        "dataset_key": list(datasets),
        "n_candidates": n_candidates,
        "sampling_temperature": sampling_temperature,
        "sampling_top_p": sampling_top_p,
    }

    for values in product(*sweeping.values()):
        params = dict(zip(list(sweeping), values))
        ilp.console.log("sweep:", params)

        config = Config(
            mode=mode,
            dataset_config=dataset_config,
            model_engine=model_engine,
            **params,
            **kwargs,
        )

        ilp.console.print("\n", str(config), "\n")

        wrapped_run(
            mode=mode,
            dataset=datasets[params["dataset_key"]],
            model=model_instance,
            config=config,
            dry_run=dry_run,
            output_prefix=output_prefix,
        )


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
