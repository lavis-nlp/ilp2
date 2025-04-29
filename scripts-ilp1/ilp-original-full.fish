#!/usr/bin/env fish

# model performance on all datasets (both all and subsampled)
# with the hyperparameters determined by the ilp-ablation-* scripts

# config

set root (dirname (status -f))

set prompt_conf generic-with-examples

echo
and . "$root"/ilp-config-model-llama3-70b.fish
and . "$root"/ilp-config-prompt-$prompt_conf.fish
and set prefix --output-prefix original-$prompt-conf-
and set datasets --datasets '*'
or return


set ilp poetry run ilp run-experiment
for filename in original-subsampled # original
    $ilp --split test $prompts $prefix $model $datasets \
        --dataset-config "$root"/../lib/irt2/conf/datasets/$filename.yaml
end
