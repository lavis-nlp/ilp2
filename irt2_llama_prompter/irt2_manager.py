from enum import Enum

import irt2
from irt2.dataset import IRT2
from irt2.dataset import MID

import textwrap
from itertools import islice
from tabulate import tabulate

from collections import Counter
from collections import defaultdict

from typing import Generator

from itertools import chain

class Scope(Enum):
    TINY = "irt2/data/irt2-cde-tiny"
    SMALL = "irt2/data/irt2-cde-tiny"
    MEDIUM = "irt2/data/irt2-cde-tiny"
    LARGE = "irt2/data/irt2-cde-tiny"

def load_irt2_data(scope):
    global data,mid2vid
    
    data = IRT2.from_dir(path=scope.value)
    
    mid2vid = {
        mid: vid
        for vid, mids in chain(
            data.closed_mentions.items(),
            data.open_mentions_val.items(),
            data.open_mentions_test.items(),
        )
        for mid in mids
    }

data = load_irt2_data(Scope.TINY)
mid2vid = {}

def uniq_rid(task):
    seen = set()
    for (mid, rid), vids in task.items():
        if rid in seen:
            continue

        seen.add(rid)
        yield (mid, rid), vids
