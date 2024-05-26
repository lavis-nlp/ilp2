#!/usr/bin/env fish

set root (dirname (status -f))


set -x ILP_DATASETS 'irt2/*'
set -x ILP_MODEL_NAME llama3-8b


for prompt_name in generic generic-with-examples
    set -x ILP_PROMPT_NAME $prompt_name
    fish $root/ilp-exp.fish
end


set -x ILP_TEXT_HEAD open.test-contexts-random-31189-30
set -x ILP_TEXT_TAIL open.test-contexts-random-31189-30
set -x ILP_PROMPT_NAME generic-with-text
fish $root/ilp-exp-with-text-irt.fish


set -x ILP_TEXT_HEAD open.test-contexts-retrieved-heads-30
set -x ILP_TEXT_TAIL open.test-contexts-retrieved-tails-30
set -x ILP_PROMPT_NAME generic-with-text
fish $root/ilp-exp-with-text-irt.fish
