#!/usr/bin/env fish

set root (dirname (status -f))

. $root/run.fish
. $root/config-dataset-irt.fish
. $root/config-model-llama.fish

set mode full-re-ranking
set dataset_split validation

set prompt_template conf/prompts/template/template-mode-3-v6.txt
set prompt_system conf/prompts/system/sysp-mode-3-to-csv-v1.yaml
set prompt_question conf/prompts/question/prompt-templates-generic-v6.yaml

run_experiments \
    --n-candidates 8 \
    --mentions-per-candidate 10 \
    $argv
