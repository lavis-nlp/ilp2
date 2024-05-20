##!/usr/bin/env fish

# model performance on all datasets (both all and subsampled)
# with the hyperparameters determined by the ilp-ablation-* scripts

set argv \
    --split test \
    --tensor-parallel-size 4 \
    --system-prompt conf/prompts/system/sysp_generic_to_json_v8.json \
    --question-template conf/prompts/question/prompt-templates-all-v1.yaml \
    --model /data/hiwi/lukas/llms/llama3/Meta-Llama-3-70B-Instruct \
    --output-prefix all-irt-1- \
    --sampling-use-beam-search n \
    --dry-run


set ilp poetry run ilp run-experiment
for filename in original original-subsampled

    set dataset --dataset-config lib/irt2/conf/datasets/$filename.yaml

    $ilp $argv $dataset \
        --datasets irt2/tiny \
        --sampling-temperature 0.4 \
        --sampling-best-of 2 \
        --sampling-top-p 1

    $ilp $argv $dataset \
        --datasets irt2/small \
        --sampling-temperature 0.4 \
        --sampling-best-of 2 \
        --sampling-top-p 1

    $ilp $argv $dataset \
        --datasets irt2/medium \
        --sampling-temperature 0.4 \
        --sampling-best-of 2 \
        --sampling-top-p 1

    $ilp $argv $dataset \
        --datasets irt2/large \
        --sampling-temperature $temperature \
        --sampling-use-beam-search n
end
