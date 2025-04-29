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
and . "$root"/ilp-config-prompt-"$ILP_PROMPT_NAME"-$ILP_PARSER.fish
and set prefix --output-prefix exp-"$ILP_PROMPT_NAME"-
and set datasets --datasets $ILP_DATASETS
and set parser --parser $ILP_PARSER
or return

set ilp poetry run ilp run-experiment # --dry-run
for filename in original-subsampled # original
    set dataset_config --dataset-config lib/irt2/conf/datasets/$filename.yaml
    for dataset in fb15k237 wn18rr wikidata5m
        $ilp --split $ILP_SPLIT $prompts $prefix $model $dataset_config \
            --datasets blp/$dataset \
            --texts-head "data/blp/text/"$dataset"/"$ILP_TEXT_HEAD".pkl" \
            --texts-tail "data/blp/text/"$dataset"/"$ILP_TEXT_TAIL".pkl"
    end
end
