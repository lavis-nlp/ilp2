from irt2_llm_prompter import run_config,model_prompter
from irt2.dataset import IRT2
from irt2.dataset import MID
import irt2

import csv

from itertools import chain

import datetime
from pathlib import Path

# TODO
# Test laufen lassen
def run_test(run_config:run_config):
    dir:Path = create_result_folder()
    run_config.export('run_config',dir)
    data:irt2.dataset.IRT2 = IRT2.from_dir(path=run_config.irt2_data_path)
    mid2vid = {
        mid: vid
        for vid, mids in chain(
            data.closed_mentions.items(),
            data.open_mentions_val.items(),
            data.open_mentions_test.items(),
        )
        for mid in mids
    }

    for (mid,rid),_ in data.open_kgc_val_tails.items():
        mention = data.mentions[mid]
        relation = data.relations[rid].split(':')[1]
        answers = model_prompter.prompt_model(run_config.get_tail_prompt(mention,relation))
        print(answers)

def uniq_rid(task):
    seen = set()
    for (mid, rid), vids in task.items():
        if rid in seen:
            continue

        seen.add(rid)
        yield (mid, rid), vids


# Ordner für Testergebnisse anlegen, Pfad zurückgeben
def create_result_folder() -> Path:
    path = Path('results')
    if not path.exists():
        path.mkdir()
    formatted_date_time = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    folder_name = 'result_{}'.format(formatted_date_time)
    dir_path = path / folder_name
    i = 0
    while dir_path.exists():
        i += 1
        formatted_date_time = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        folder_name = 'result_{}_({})'.format(formatted_date_time,i)
        dir_path = path / folder_name
    dir_path.mkdir()
    return dir_path