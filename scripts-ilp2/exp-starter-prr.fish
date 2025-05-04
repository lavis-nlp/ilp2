#!/usr/bin/env fish

set root (dirname (status -f))

. $root/run.fish
. $root/config-dataset-irt.fish
. $root/config-model-llama.fish
. $root/config-mode-prompt-re-ranking.fish

set dataset_split validation

run_experiments $argv
