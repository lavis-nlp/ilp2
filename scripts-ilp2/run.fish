function params -a name
    echo $name (string join -- " $name " $argv[2..]) | tr ' ' \n
end


function run_experiments
    poetry run ilp $debug run-experiment \
        --mode $mode \
        # set by . config-dataset-*.fish
        --dataset-config $dataset_config \
        (params --dataset-key $dataset_keys) \
        --dataset-split $split \
        # set by . config-model-*.fish
        --model-path $model_path \
        --model-tensor-parallel-size $model_tensor_parallel_size \
        --model-gpu-memory-utilization $model_gpu_memory_utilization \
        --model-parser $model_parser \
        --model-engine $model_engine \
        --model-dtype $model_dtype \
        $model_quantization \
        # prompt related
        --prompt-template $prompt_template \
        --system-prompt $system_prompt \
        --question-template $question_template \
        $argv
end
