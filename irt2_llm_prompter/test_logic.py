from irt2_llm_prompter import run_config, model_prompter
from irt2_llm_prompter.run_config import RunConfig
from irt2_llm_prompter.model_prompter import Model

from irt2_llm_prompter.sample_config import all_config

from vllm import RequestOutput, SamplingParams
from typing import List, Literal

import re
import json

from irt2.dataset import IRT2
from irt2.types import Split, Task, Sample, MID, RID, VID
from irt2.evaluation import evaluate

import pickle

import datetime
from pathlib import Path

Tasks = dict[tuple[MID, RID], set[VID]]


def run_on_val(
    run_config: RunConfig,
):
    _run_benchmark(run_config=run_config, mode="validation")


def run_on_test(
    run_config: RunConfig,
):
    _run_benchmark(run_config=run_config, mode="test")


# TODO
# Test laufen lassen
def _run_benchmark(
    run_config: RunConfig,
    mode: Literal["validation", "test"],
):
    """Testet Kombination aus RunConfig und Model und erstellt Evaluation"""
    # Erstellt Output-Ordner
    dir: Path = create_result_folder()
    # Exportiert Run-Config in Output-Ordner
    run_config.export("RunConfig", dir)
    dataset = all_config["datasets"][run_config.data_type]
    data: IRT2 = IRT2.from_dir(path=dataset["path"])

    print(run_config)

    sampling_params = sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        use_beam_search=True,
        best_of=2,
        max_tokens=4098,
    )

    # Model
    model = model_prompter.Model(
        path=run_config.model_path,
        params=sampling_params,
        tensor_parallel_size=run_config.tensor_parallel_size,
    )
    model.load_model()

    # Subsampling
    percentage = dataset["percentage"][mode]
    seed = all_config["seed"]
    data = data.tasks_subsample_kgc(percentage=percentage, seed=seed)

    if mode == "validation":
        tail_predictions = create_Predictions(
            tasks=data.open_kgc_val_tails,
            splits=(Split.train, Split.valid),
            ds=data,
            prompt_templates=run_config.tail_templates,
            system_prompt=run_config.system_prompt,
            model=model,
        )
        head_predictions = create_Predictions(
            tasks=data.open_kgc_val_heads,
            splits=(Split.train, Split.valid),
            ds=data,
            prompt_templates=run_config.head_templates,
            system_prompt=run_config.system_prompt,
            model=model,
        )
    else:
        tail_predictions = create_Predictions(
            tasks=data.open_kgc_test_tails,
            splits=(Split.train, Split.valid, Split.test),
            ds=data,
            prompt_templates=run_config.tail_templates,
            system_prompt=run_config.system_prompt,
            model=model,
        )
        head_predictions = create_Predictions(
            tasks=data.open_kgc_test_heads,
            splits=(Split.train, Split.valid, Split.test),
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
        split=mode,
        head_predictions=head_predictions,
        tail_predictions=tail_predictions,
    )
    print(evaluation)

    json_eval = json.dumps(evaluation)

    result_file = open(dir / "result.txt", "w")
    result_file.write(json_eval)
    result_file.write("\n")

    with open(dir / "result.pkl", "wb") as file:
        pickle.dump(result, file)


def parseAnswer(
    model_response: List[RequestOutput],
):
    """Parsed Model-Output zu Antwortliste"""
    # Extrahiert Text aus Output, stript ihn
    result = model_response[0].outputs[0].text.strip()
    # Trennt pre- oder suffixe
    result = re.search(r"\{([^{}]*)\}", result)
    if result is None:
        return {}
    result = result.group(0)
    # Parsed zu JSON-File
    json_response = json.loads(result)
    # Extrahiert Antwort Liste
    return json_response["answer"]


def create_Predictions(
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

    for (mid, rid), gt_vids in tasks.items():
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

            mentions = set(s.strip().lower() for s in parseAnswer(response))

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

        predictions.append(((mid, rid), [(vid, 1) for vid in pr_vids]))

    return predictions


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
        return templates["generic"].format(mention, relation)
    else:
        return templates[relation].format(mention)


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
