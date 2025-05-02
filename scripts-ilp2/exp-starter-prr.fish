#!/usr/bin/env fish

set root (dirname (status -f))

. $root/config-dataset-irt.fish
. $root/config-model-llama.fish

set split validation
set mode prompt-re-ranking

set prompt_template conf/prompts/template/template-mode-2-v2.1.txt
set system_prompt conf/prompts/system/sysp-mode-2-to-csv-v2.1.yaml
set question_template conf/prompts/question/prompt-templates-generic-v6.yaml

. $root/run.fish

run_experiments \
    --n-candidates 8 \
    --mentions-per-candidate 10 \
    $argv
