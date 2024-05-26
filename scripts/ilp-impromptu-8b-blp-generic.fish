#!/usr/bin/env fish

set root (dirname (status -f))


set -x ILP_DATASETS 'blp/*'
set -x ILP_MODEL_NAME llama3-8b


for prompt_name in generic generic-with-examples
    set -x ILP_PROMPT_NAME $prompt_name
    fish $root/ilp-exp.fish
end


set -x ILP_TEXT_HEAD text
set -x ILP_TEXT_TAIL text
set -x ILP_PROMPT_NAME generic-with-text
fish $root/ilp-exp-with-text-blp.fish
