from irt2_llm_prompter import run_config, model_prompter
from irt2_llm_prompter.run_config import RunConfig

import irt2

from irt2.dataset import IRT2
from irt2.types import Split, Task, Sample, MID, RID, VID

import csv
import pickle

from itertools import chain, islice

import datetime
from pathlib import Path

Tasks = dict[tuple[MID, RID], set[VID]]


# TODO
# Test laufen lassen
def run_test(run_config: RunConfig):
    dir: Path = create_result_folder()
    run_config.export("RunConfig", dir)
    data: irt2.dataset.IRT2 = IRT2.from_dir(path=run_config.irt2_data_path)

    tail_predictions = create_Predictions(
        tasks=data.open_kgc_val_tails,
        ds=data,
        run_config=run_config,
    )

    head_predictions = create_Predictions(
        tasks=data.open_kgc_val_heads,
        ds=data,
        run_config=run_config,
    )

    result = dict()
    result["tail_predictions"] = tail_predictions
    result["head_predictions"] = head_predictions

    with open(dir / "result.pkl", "wb") as file:
        pickle.dump(result, file)

    # for (mid, rid), _ in data.open_kgc_val_tails.items():
    #    mention = data.mentions[mid]
    #    relation = data.relations[rid].split(":")[1]
    #    answers = model_prompter.prompt_model(
    #       run_config.get_tail_prompt(mention, relation)
    #   )
    #   print(answers)


def create_Predictions(tasks: Tasks, ds: IRT2, run_config: RunConfig, **_) -> list:
    splits = (Split.train, Split.valid)
    ids = ds.idmap

    predictions = []

    for (mid, rid), gt_vids in islice(tasks.items(), 1):
        mention = ds.mentions[mid]
        relation = ds.relations[rid].split(":")[1]
        response = model_prompter.prompt_model(
            run_config.get_tail_prompt(mention, relation)
        )
        print(response)
        mentions = set(s.strip().lower() for s in response.split("|"))
        print(mentions,'')
        pr_vids = ds.find_by_mention(
            *mentions,
            splits=splits,
        )

        predictions.append(((mid, rid), [(vid, 1) for vid in pr_vids]))

    return predictions


# Ordner für Testergebnisse anlegen, Pfad zurückgeben
def create_result_folder() -> Path:
    """Erstellt Ordner für Run-Result"""
    path = Path("results")
    if not path.exists():
        path.mkdir()
    formatted_date_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    folder_name = "result_{}_{}".format(len(list(path.glob("*/"))), formatted_date_time)
    dir_path = path / folder_name
    return dir_path
