#!/usr/bin/env fish

set root (dirname (status -f))

set -x ILP_MODEL_NAME llama3-70b
set -x ILP_PARSER csv
set -x ILP_SPLIT test
set -x ILP_PENALITY 1

# BLP GENERIC TEXT

set -x ILP_DATASETS 'blp/*'
set -x ILP_TEXT_HEAD text
set -x ILP_TEXT_TAIL text
set -x ILP_PROMPT_NAME generic-with-text
fish $root/ilp-exp-with-text-blp.fish