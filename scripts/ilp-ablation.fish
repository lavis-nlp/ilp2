#!/usr/bin/env fish

# ablation study observing model performance based
# on temperature, beam width and top-p


# defs

set model_llama3_8b \
    --model data/llms/llama3/Meta-Llama-3-8B-Instruct \
    --tensor-parallel-size 1

set model_llama3_70b \
    --model data/llms/llama3/Meta-Llama-3-70B-Instruct \
    --tensor-parallel-size 4

# config

set model $model_llama3_8b
set datasets irt2/tiny
set prefix abl-irt2-tiny-2-

set prompts \
    --system-prompt conf/prompts/system/sysp-to-json-v1.yaml \
    --question-template conf/prompts/question/prompt-templates-all-v1.yaml \
    --prompt-template conf/prompts/template/template-without-text-v1.txt


# --

set argv $model $prompts \
    --dataset-config lib/irt2/conf/datasets/original-subsampled.yaml \
    --datasets $datasets \
    --split validation \
    --output-prefix $prefix


set ilp poetry run ilp run-experiment


# beam search
for early_stopping in y n
    for best_of in 2 4 6 8
        echo -e '\n====================\n'

        $ilp $argv \
            --sampling-early-stopping $early_stopping \
            --sampling-best-of $best_of \
            --sampling-use-beam-search y

    end
end


# random sampling
for temperature in 0.1 0.2 0.3 0.4 0.5 0.6 0.7
    for top_p in 1
        for best_of in 1 2 4
            echo -e '\n====================\n'

            $ilp $argv \
                --sampling-temperature $temperature \
                --sampling-top-p $top_p \
                --sampling-best-of $best_of \
                --sampling-use-beam-search n
        end
    end
end


# greedy
echo -e '\n====================\n'
$ilp $argv \
    --sampling-use-beam-search n \
    --sampling-temperature 0 \
    --sampling-best-of 1
