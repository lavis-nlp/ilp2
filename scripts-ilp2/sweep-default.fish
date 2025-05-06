#!/usr/bin/env fish
# grid search for random sampling

set root (dirname (status -f))
. $root/run.fish

set debug # -qd

# recommended parameters for LLAMA are
#   - temperature: 0.6
#   - top_p: 0.9
. $root/config-model-llama.fish

# recommended parameters for LLAMA are
#   - temperature: 0.6
#   - top_p: 0.95
# . $root/config-model-deepseek.fish

set dataset_config lib/irt2/conf/datasets/original-subsampled.yaml
set dataset_keys irt2/tiny # irt2/small irt2/medium irt2/large
set dataset_split validation

# . $root/config-mode-default.fish
. $root/config-mode-prompt-re-ranking.fish
# . $root/config-mode-full-re-ranking.fish

# greedy sampling
# top_p is 1 because temp is 0
# run_experiments \
#     --sampling-temperature 0.0 \
#     --sampling-top-p 1.0 \
#     --sampling-use-beam-search no \
#     --sampling-early-stopping yes \
#     --sampling-repetition-penalty 1 \
#     $argv

set temperatures (params --sampling-temperature (seq 0.1 0.1 0.9))
set top_ps (params --sampling-top-p (seq 0.2 0.1 0.9))
set rnd (tr -dc A-Za-z0-9 </dev/urandom | head -c 5; echo)

run_experiments \
    $temperatures \
    $top_ps \
    --model-use-beam-search no \
    --sampling-early-stopping yes \
    --sampling-repetition-penalty 1 \
    --output-prefix sweep-$rnd- \
    $argv
