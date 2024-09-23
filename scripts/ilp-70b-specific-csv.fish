#!/usr/bin/env fish

set root (dirname (status -f))

set -x ILP_MODEL_NAME llama3-70b


# IRT2 SPECIFIC

set -x ILP_DATASETS 'irt2/*'
set -x ILP_PROMPT_NAME specific
fish $root/ilp-exp-70b-specific.fish
