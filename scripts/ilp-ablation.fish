#!/usr/bin/env fish

# ablation study observing model performance based
# on temperature, beam width and top-p


set argv \
    --tensor-parallel-size 4 \
    --dataset-config lib/irt2/conf/datasets/original-subsampled.yaml \
    --datasets irt2/tiny \
    --system-prompt conf/prompts/system/sysp_generic_to_json_v8.json \
    --question-template conf/prompts/question/prompt_templates_large_v5.json \
    --split validation \
    --dataset-config lib/irt2/conf/datasets/original-subsampled.yaml \
    --model /data/hiwi/lukas/llms/llama3/Meta-Llama-3-70B-Instruct \
    --output-prefix abl-1-

set ilp poetry run ilp run-experiment


# beam search
for early_stopping in y n
    for best_of in 2 4
        echo -e '\n====================\n'

        $ilp $argv \
            --sampling-early-stopping $early_stopping \
            --sampling-best-of $best_of \
            --sampling-use-beam-search y
    end
end


# random sampling
for temperature in 0.5 1 1.5
    for top_p in 0.5 1
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
