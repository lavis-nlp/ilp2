#!/usr/bin/env fish

set root (dirname (status -f))
set debug # -qd

. $root/config-dataset-irt.fish
. $root/config-model-deepseek.fish
# . $root/config-model-llama.fish

set split validation
set mode default

set prompt_template conf/prompts/template/template-ripe-generic-v3.txt
set system_prompt conf/prompts/system/sysp-to-csv-v4.1.yaml
set question_template conf/prompts/question/prompt-templates-generic-v6.yaml

set n_cands 0 # 5 10 15 20
set mentions_per_candidate 10

. $root/run.fish
run_experiments $argv
