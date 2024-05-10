from irt2_llm_prompter import run_config, model_prompter
from irt2_llm_prompter.run_config import RunConfig
from irt2_llm_prompter.model_prompter import Model

from vllm import RequestOutput
from typing import List

import re
import json

import irt2

from irt2.dataset import IRT2
from irt2.types import Split, Task, Sample, MID, RID, VID
from irt2.evaluation import evaluate

import pickle

import datetime
from pathlib import Path

Tasks = dict[tuple[MID, RID], set[VID]]


# TODO
# Test laufen lassen
def run_test(run_config: RunConfig, model: Model):
    """Testet Kombination aus RunConfig und Model und erstellt Evaluation"""
    # Erstellt Output-Ordner
    dir: Path = create_result_folder()
    # Exportiert Run-Config in Output-Ordner
    run_config.export("RunConfig", dir)
    data: IRT2 = IRT2.from_dir(path=run_config.irt2_data_path)

    # Subsampling
    at_most = 1000
    seed = 31189
    data = data.tasks_subsample(to=at_most, seed=seed)

    #print(data.open_kgc_val_tails)

    tail_predictions = create_Predictions(
        tasks=data.open_kgc_val_tails,
        ds=data,
        prompt_templates=run_config.tail_templates,
        system_prompt=run_config.system_prompt,
        model=model,
    )

    head_predictions = create_Predictions(
        tasks=data.open_kgc_val_heads,
        ds=data,
        prompt_templates=run_config.head_templates,
        system_prompt=run_config.system_prompt,
        model=model,
    )

    result = dict()
    result["tail_predictions"] = tail_predictions
    result["head_predictions"] = head_predictions

    evaluation = evaluate(
        ds=data,
        task="kgc",
        split="validation",
        head_predictions=head_predictions,
        tail_predictions=tail_predictions,
    )
    print(evaluation)

    result_file = open(dir / result.txt,"w")
    result_file.write(evaluation)

    with open(dir / "result.pkl", "wb") as file:
        pickle.dump(result, file)


def parseAnswer(model_response: List[RequestOutput]):
    """Parsed Model-Output zu Antwortliste"""
    # Extrahiert Text aus Output, stript ihn
    result = model_response[0].outputs[0].text.strip()
    # Entfernt pre- oder suffixe
    result = re.search(r"\{([^{}]*)\}", result).group(0)
    # Parsed zu JSON-File
    json_response = json.loads(result)
    # Extrahiert Antwort Liste
    return json_response["answer"]


def create_Predictions(
    tasks: Tasks, ds: IRT2, system_prompt: str, prompt_templates: dict, model: Model
) -> list:
    """Erstellt prediction Objekt für evaluation-Methode"""
    splits = (Split.train, Split.valid)
    ids = ds.idmap

    predictions = []

    for (mid, rid), gt_vids in tasks.items():
        mention = ds.mentions[mid]
        relation = ds.relations[rid].split(":")[1]
        prompt = build_prompt(
            system_prompt=system_prompt,
            templates=prompt_templates,
            mention=mention,
            relation=relation,
        )
        #print("-" * 20)
        #print("Prompt: ", prompt)

        response = model.prompt_model(prompt=prompt)

        if response[0].outputs[0].text:
            #print("Antwort: ", response[0].outputs[0].text)

            mentions = set(s.strip().lower() for s in parseAnswer(response))

            #print("Extrakt: ", mentions)

            pr_vids = ds.find_by_mention(
                *mentions,
                splits=splits,
            )

            #print("Model-VIDs: ", ((mid, rid), [(vid, 1) for vid in pr_vids]))

            #print_ground_truth(mid, rid, gt_vids, ds)
        else:
            pr_vids = set()

        predictions.append(((mid, rid), [(vid, 1) for vid in pr_vids]))

    #print(predictions)
    return predictions


def print_ground_truth(mid, rid, gt_vids, ds: IRT2):
    """Logging Funktion, True Vids und True Mentions"""
    print("True vids: ", (mid, rid), end="")
    for vertex in ((vid, 1) for vid in gt_vids):
        print(vertex, end=", ")
    print("")
    print("True vids: (" + ds.mentions[mid] + "," + ds.relations[rid] + ") -> ", end="")
    for vertex in (ds.vertices[vid] for vid in gt_vids):
        print(vertex, end=", ")
    print("")
    ids = ds.idmap
    splits = (Split.train, Split.valid)
    mentions = {
        ids.mid2str[mid] for mids in map(ids.vid2mids.get, gt_vids) for mid in mids
    }

    pr_vids = ds.find_by_mention(
        *mentions,
        splits=splits,
    )

    print("True mentions: ", (mid, rid), end="")
    for vertex in ((vid, 1) for vid in pr_vids):
        print(vertex, end=", ")
    print("")
    print(
        "True mentions: (" + ds.mentions[mid] + "," + ds.relations[rid] + ") -> ",
        end="",
    )
    for vertex in (ds.vertices[vid] for vid in pr_vids):
        print(vertex, end=", ")
    print("")


def build_prompt(
    system_prompt: str, templates: dict, mention: str, relation: str
) -> str:
    """Baut Prompt zusammen"""
    if relation not in templates:
        #print("Used generic")
        content = templates["generic"].format(mention, relation)
    else:
        content = templates[relation].format(mention)

    prompt = "{} {}".format(system_prompt, content)
    return prompt

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
