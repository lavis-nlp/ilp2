#!/usr/bin/env fish

# model performance on all datasets (both all and subsampled)
# with the hyperparameters determined by the ilp-ablation-* scripts

# config

set root (dirname (status -f))
if [ -z "$ILP_MODEL_NAME" -o -z "$ILP_PROMPT_NAME" -o -z "$ILP_DATASETS" ]
    return 2
end


echo
and . "$root"/ilp-config-model-"$ILP_MODEL_NAME".fish
and . "$root"/ilp-config-prompt-"$ILP_PROMPT_NAME".fish
and set prefix --output-prefix exp-"$ILP_PROMPT_NAME"-
and set datasets --datasets $ILP_DATASETS
or return

set ilp poetry run ilp run-experiment # --dry-run
for filename in original-subsampled # original
    $ilp --split test $prompts $prefix $model $datasets \
        --dataset-config "$root"/../lib/irt2/conf/datasets/$filename.yaml
end
