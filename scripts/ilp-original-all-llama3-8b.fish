#!/usr/bin/env fish

# model performance on all datasets (both all and subsampled)
# with the hyperparameters determined by the ilp-ablation-* scripts

set argv \
    --split test \
    --tensor-parallel-size 1 \
    --model /data/hiwi/lukas/llms/llama3/Meta-Llama-3-8B-Instruct \
    --prompt-template conf/prompts/template/template-without-text-v1.txt \
    --system-prompt conf/prompts/system/sysp-to-json-v1.yaml \
    --question-template conf/prompts/question/prompt-templates-all-v1.yaml \
    --output-prefix all-irt-2- \
    --sampling-use-beam-search n


set ilp poetry run ilp run-experiment
for filename in original-subsampled original

    set dataset --dataset-config lib/irt2/conf/datasets/$filename.yaml

    $ilp $argv $dataset \
        --datasets irt2/tiny \
        --sampling-temperature 0.6 \
        --sampling-best-of 2 \
        --sampling-top-p 1

    $ilp $argv $dataset \
        --datasets irt2/small \
        --sampling-temperature 0.4 \
        --sampling-best-of 2 \
        --sampling-top-p 1

    $ilp $argv $dataset \
        --datasets irt2/medium \
        --sampling-temperature 0.6 \
        --sampling-best-of 2 \
        --sampling-top-p 1

    $ilp $argv $dataset \
        --datasets irt2/large \
        --sampling-temperature 0.5 \
        --sampling-best-of 2 \
        --sampling-top-p 1

end
