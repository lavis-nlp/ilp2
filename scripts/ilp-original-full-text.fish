#!/usr/bin/env fish

# model performance on all datasets (both all and subsampled)
# with the hyperparameters determined by the ilp-ablation-* scripts

# defs

set model_llama3_8b \
    --model data/llms/llama3/Meta-Llama-3-8B-Instruct \
    --tensor-parallel-size 1 \
    --sampling-use-beam-search y \
    --sampling-early-stopping y \
    --sampling-best-of 4

set model_llama3_70b \
    --model data/llms/llama3/Meta-Llama-3-70B-Instruct \
    --tensor-parallel-size 4 \
    --sampling-use-beam-search y \
    --sampling-early-stopping y \
    --sampling-best-of 2

# config

set model $model_llama3_8b
set prefix original-all-text-

set argv \
    --split test \
    --prompt-template conf/prompts/template/template-with-text-v1.txt \
    --system-prompt conf/prompts/system/sysp-to-json-v1.yaml \
    --question-template conf/prompts/question/prompt-templates-all-v1.yaml \
    --output-prefix $prefix


set ilp poetry run ilp run-experiment
for filename in original-subsampled # original
    set dataset_config --dataset-config lib/irt2/conf/datasets/$filename.yaml
    for dataset in tiny small medium large
        $ilp $argv $dataset_config $model --datasets irt2/$dataset \
            --texts "lib/irt2/data/irt2/irt2-cde-"$dataset"/open.test-contexts-31189-30.pkl"
    end
end
