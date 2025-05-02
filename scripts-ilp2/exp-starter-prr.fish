#!/usr/bin/env fish

set root (dirname (status -f))

. $root/run.fish
. $root/config-dataset-irt.fish
. $root/config-model-llama.fish

set dataset_split validation
set mode prompt-re-ranking

set prompt_template conf/prompts/template/template-mode-2-v2.1.txt
set prompt_system conf/prompts/system/sysp-mode-2-to-csv-v2.1.yaml
set prompt_question conf/prompts/question/prompt-templates-generic-v6.yaml

run_experiments \
    --n-candidates 8 \
    --mentions-per-candidate 10 \
    $argv
