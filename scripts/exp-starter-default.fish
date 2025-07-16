#!/usr/bin/env fish

set root (dirname (status -f))
set debug # -qd

. $root/run.fish
. $root/config-dataset-irt.fish
. $root/config-model-deepseek.fish
# . $root/config-model-llama.fish
. $root/config-mode-default.fish

set dataset_split validation

run_experiments $argv \
    --sampling-temperature 0.6 \
    --sampling-top-p 0.95
