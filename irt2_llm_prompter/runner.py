import datetime
import json
import pickle
import re
from itertools import islice
from pathlib import Path
from traceback import print_exc
from typing import List, Literal

from irt2.dataset import IRT2
from irt2.evaluation import evaluate
from irt2.types import MID, RID, VID, Sample, Split, Task
from ktz.collections import path
from vllm import RequestOutput, SamplingParams

import irt2_llm_prompter as ilp
from irt2_llm_prompter import model_prompter, run_config
from irt2_llm_prompter.model_prompter import Model
from irt2_llm_prompter.run_config import RunConfig
from irt2_llm_prompter.sample_config import all_config

Tasks = dict[tuple[MID, RID], set[VID]]


# formerly: test_run.py:_run_benchmark
def run(
    dataset: IRT2,
    run_config: RunConfig,
    sampling_params: SamplingParams,
    result_folder: str | Path,
):
    """Testet Kombination aus RunConfig und Model und erstellt Evaluation"""

    out = path(result_folder, create=True)
    run_config.save(to=out / "run-config.yaml")

    model = model_prompter.Model(
        path=run_config.model_path,
        params=sampling_params,
        tensor_parallel_size=run_config.tensor_parallel_size,
    )

    # model.load_model()

    tail_tasks, head_tasks = {
        "validation": (
            dataset.open_kgc_val_tails,
            dataset.open_kgc_val_heads,
        ),
        "test": (
            dataset.open_kgc_test_tails,
            dataset.open_kgc_test_heads,
        ),
    }[run_config.split]

    ilp.console.log("creating tail predictions")
    tail_predictions = create_predictions(
        tasks=tail_tasks,
        splits=(Split.train,),
        ds=dataset,
        prompt_templates=run_config.tail_templates,
        system_prompt=run_config.system_prompt,
        model=model,
    )

    ilp.console.log("creating head predictions")
    head_predictions = create_predictions(
        tasks=head_tasks,
        splits=(Split.train,),
        ds=dataset,
        prompt_templates=run_config.head_templates,
        system_prompt=run_config.system_prompt,
        model=model,
    )

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


def parse_answer(
    model_response: List[RequestOutput],
) -> list[str]:
    """Parsed Model-Output zu Antwortliste"""
    # Extrahiert Text aus Output, stript ihn
    result = model_response[0].outputs[0].text.strip()

    # Trennt pre- oder suffixe
    if "{" not in result or "}" not in result:
        return []

    sub = result[result.index("{") : result.index("}") + 1]

    # Extrahiert Antwort Liste
    return json.loads(sub)["answer"]


def create_predictions(
    tasks: Tasks,
    splits: tuple,
    ds: IRT2,
    system_prompt: str,
    prompt_templates: dict,
    model: Model,
) -> list:
    """Erstellt prediction Objekt für evaluation-Methode"""
    ids = ds.idmap

    predictions = []
    n = 1

    for (mid, rid), gt_vids in islice(tasks.items(), None):
        try:
            prediction = create_prediction(
                mid=mid,
                rid=rid,
                prompt_templates=prompt_templates,
                system_prompt=system_prompt,
                ds=ds,
                splits=splits,
                model=model,
                gt_vids=gt_vids,
            )
        except Exception as e:
            print_exc()
            predictions.append(((mid, rid), []))
            continue

        predictions.append(((mid, rid), [(vid, 1) for vid in prediction]))

    return predictions


def create_prediction(
    mid: MID,
    rid: RID,
    prompt_templates: dict,
    system_prompt: str,
    ds: IRT2,
    splits: tuple,
    model: model_prompter.Model,
    gt_vids: set[VID],
) -> set[VID]:
    mention = ds.mentions[mid]
    relation = ds.relations[rid].split(":")[1]

    prompt_body = build_prompt_body(
        templates=prompt_templates,
        mention=mention,
        relation=relation,
    )

    prompt = "{} {}".format(system_prompt, prompt_body)
    print("----------------------")
    print("Task: {}, {} -> {}\n".format(mention, relation, prompt_body))

    response = model.prompt_model(prompt=prompt)

    if response[0].outputs[0].text:
        print("Antwort: ", response[0].outputs[0].text.strip())

        mentions = set(s.strip().lower() for s in parse_answer(response))

        print("Extrakt: ", mentions)

        pr_vids = ds.find_by_mention(
            *mentions,
            splits=splits,
        )

        print(
            "Gefundene Vertices: ",
            [ds.vertices[vid].split(":")[1] for vid in pr_vids],
        )

        check_ground_truth(
            mid=mid,
            rid=rid,
            gt_vids=gt_vids,
            pr_vids=pr_vids,
            ds=ds,
        )
    else:
        pr_vids = set()
        print("Leere Antwort!")

    return pr_vids


def check_ground_truth(
    mid,
    rid,
    gt_vids,
    pr_vids,
    ds: IRT2,
):
    """Logging Funktion, vergleiche mit gt"""

    print("GT-Mentions (-FOUND): ", end="")
    for mention in [
        ds.idmap.mid2str[mid] for vid in gt_vids for mid in ds.idmap.vid2mids[vid]
    ]:
        if mention not in pr_vids:
            print(mention, end=", ")
    print("\n")

    print("True vids: ", end="")
    for vertex in (ds.vertices[vid] for vid in gt_vids):
        print(vertex.split(":")[1], end=", ")
    print("\n")

    n: int = 0
    for vid in gt_vids:
        if vid in pr_vids:
            n += 1
    print("Korrekt: {}/{}\n".format(n, len(gt_vids)))


def build_prompt_body(
    templates: dict,
    mention: str,
    relation: str,
) -> str:
    """Baut Prompt-Body zusammen"""
    if relation not in templates:
        # print("Used generic")
        return templates["generic"].format(m=mention, r=relation)
    else:
        return templates[relation].format(m=mention)


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
