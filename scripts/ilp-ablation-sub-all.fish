#!/usr/bin/env fish

# ablation study observing model performance based
# on temperature, beam width and top-p


set argv \
    --tensor-parallel-size 4 \
    --dataset-config lib/irt2/conf/datasets/original-subsampled.yaml \
    --datasets 'irt2/*' \
    --system-prompt conf/prompts/system/sysp_generic_to_json_v8.json \
    --question-template conf/prompts/question/prompt-templates-all-v1.yaml \
    --split validation \
    --model /data/hiwi/lukas/llms/llama3/Meta-Llama-3-70B-Instruct \
    --output-prefix abl-3-


set ilp poetry run ilp run-experiment


for temperature in 0.1 0.2 0.3 0.4 0.5
    echo -e "\n==================== temperature = $temperature\n"

    $ilp $argv \
        --sampling-temperature $temperature \
        --sampling-use-beam-search n

end
