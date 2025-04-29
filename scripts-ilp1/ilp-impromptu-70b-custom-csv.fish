#!/usr/bin/env fish

set root (dirname (status -f))

set -x ILP_MODEL_NAME llama3-70b
set -x ILP_PARSER csv
set -x ILP_SPLIT validation
set -x ILP_PENALITY 1.5

# IRT2 SPECIFIC

#set -x ILP_DATASETS 'irt2/*'
#for prompt_name in specific-with-examples # specific
#    set -x ILP_PROMPT_NAME $prompt_name
#    fish $root/ilp-exp.fish


#set -x ILP_PROMPT_NAME specific-with-text
#set -x ILP_TEXT_HEAD open.test-contexts-random-31189-30
#set -x ILP_TEXT_TAIL open.test-contexts-random-31189-30
#fish $root/ilp-exp-with-text-irt.fish

#set -x ILP_PROMPT_NAME specific-with-text
#set -x ILP_TEXT_HEAD open.test-contexts-retrieved-heads-30
#set -x ILP_TEXT_TAIL open.test-contexts-retrieved-tails-30
#fish $root/ilp-exp-with-text-irt.fish


# ALL GENERIC

set -x ILP_DATASETS '*'
for prompt_name in generic #generic-with-examples
    set -x ILP_PROMPT_NAME $prompt_name
    fish $root/ilp-exp.fish
end

exit

# IRT2 GENERIC TEXT

#set -x ILP_DATASETS 'irt2/*'
#set -x ILP_TEXT_HEAD open.test-contexts-random-31189-30
#set -x ILP_TEXT_TAIL open.test-contexts-random-31189-30
#set -x ILP_PROMPT_NAME generic-with-text
#fish $root/ilp-exp-with-text-irt.fish


#set -x ILP_TEXT_HEAD open.test-contexts-retrieved-heads-30
#set -x ILP_TEXT_TAIL open.test-contexts-retrieved-tails-30
#set -x ILP_PROMPT_NAME generic-with-text
#fish $root/ilp-exp-with-text-irt.fish


# BLP GENERIC TEXT

#set -x ILP_DATASETS 'blp/*'
#set -x ILP_TEXT_HEAD text
#set -x ILP_TEXT_TAIL text
#set -x ILP_PROMPT_NAME generic-with-text
#fish $root/ilp-exp-with-text-blp.fish
