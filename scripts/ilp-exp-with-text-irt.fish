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
    set dataset_config --dataset-config lib/irt2/conf/datasets/$filename.yaml
    for dataset in tiny small medium large
        $ilp --split test $prompts $prefix $model $dataset_config \
            --datasets irt2/$dataset \
            --texts-head "data/irt2/irt2-cde-"$dataset"/"$ILP_TEXT_HEAD".pkl" \
            --texts-tail "data/irt2/irt2-cde-"$dataset"/"$ILP_TEXT_TAIL".pkl"
    end
end
