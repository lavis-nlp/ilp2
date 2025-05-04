#!/usr/bin/env fish

set root (dirname (status -f))

. $root/run.fish
. $root/config-dataset-irt.fish
. $root/config-model-llama.fish
. $root/config-mode-ranker-results.fish

set split validation

run_experiments $argv
