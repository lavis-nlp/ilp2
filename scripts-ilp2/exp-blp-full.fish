#!/usr/bin/env fish
#
# run all experiments with final parameter configuration as determined
# by experiments detailed in the spreadsheet (see readme)

set dataset_split validation

# setup

set root (dirname (status -f))

. $root/run.fish
. $root/config-dataset-blp.fish


# PRE-RANKER

# mode: pre-ranker
# . $root/config-mode-pre-ranker.fish
# set dataset_keys irt2/tiny irt2/small irt2/medium irt2/large
# . $root/config-model-llama.fish # set any model, does not matter here
# run_experiments $argv


# VANILLA LLAMA

. $root/config-model-llama.fish

# mode: default
. $root/config-mode-default.fish

set debug -qd
run_experiments $argv --dry-run

# mode: prompt-reranking
# . $root/config-mode-prompt-re-ranking.fish
# set dataset_keys irt2/tiny irt2/small irt2/medium irt2/large
# run_experiments $argv


# mode: full-reranking
# . $root/config-mode-full-re-ranking.fish
# set dataset_keys irt2/tiny irt2/small irt2/medium irt2/large
# run_experiments $argv


# TODO add BLP

# DEEPSEEK LLAMA

# . $root/config-model-deepseek.fish

# # mode: default
# . $root/config-mode-default.fish
# set dataset_keys irt2/tiny irt2/small irt2/medium irt2/large
# run_experiments $argv $default_temperature $default_top_p

# # mode: prompt-reranking
# . $root/config-mode-prompt-reranking.fish
# set dataset_keys irt2/tiny irt2/small irt2/medium irt2/large
# run_experiments $argv $default_temperature $default_top_p


# # mode: full-reranking
# . $root/config-mode-full-reranking.fish
# set dataset_keys irt2/tiny irt2/small irt2/medium irt2/large
# run_experiments $argv $default_temperature $default_top_p


# TODO add BLP
