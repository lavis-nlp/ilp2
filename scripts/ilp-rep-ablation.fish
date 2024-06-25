#!/usr/bin/env fish

# ablation study observing model performance based
# on temperature, beam width and top-p


# defs

set model_llama3_8b \
    --model /data/hiwi/lukas/llms/llama3/Meta-Llama-3-8B-Instruct \
    --tensor-parallel-size 1

set model_llama3_70b \
    --model /data/hiwi/lukas/llms/llama3/Meta-Llama-3-70B-Instruct \
    --tensor-parallel-size 4

# config

set model $model_llama3_70b
set datasets irt2/tiny 
set prefix lukas-abl-irt2-tiny

set prompts \
    --system-prompt conf/prompts/system/sysp-to-json-v1.yaml \
    # --question-template conf/prompts/question/prompt-templates-all-v1.yaml \
    --question-template conf/prompts/question/prompt-templates-generic-v1.yaml \
    # --prompt-template conf/prompts/template/template-without-text-v1.txt
    --prompt-template conf/prompts/template/template-generic-v1.txt


# --

set argv $model $prompts \
    --dataset-config lib/irt2/conf/datasets/original-subsampled.yaml \
    --datasets $datasets \
    --split validation \
    --output-prefix $prefix


set ilp poetry run ilp run-experiment

for penalty in 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9
    echo -e '\n====================\n'
    $ilp $argv \
	--sampling-repetition-penalty $penalty
end

