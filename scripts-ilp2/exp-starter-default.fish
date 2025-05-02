#!/usr/bin/env fish

set root (dirname (status -f))
set debug # -qd

. $root/run.fish
. $root/config-dataset-irt.fish
# . $root/config-model-deepseek.fish
. $root/config-model-llama.fish

set mode default
set dataset_split validation

set prompt_template conf/prompts/template/template-ripe-generic-v3.txt
set prompt_system conf/prompts/system/sysp-to-csv-v4.1.yaml
set prompt_question conf/prompts/question/prompt-templates-generic-v6.yaml

run_experiments $argv
