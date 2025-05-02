function params -a name
    echo $name (string join -- " $name " $argv[2..]) | tr ' ' \n
end


function run_experiments
    poetry run ilp $debug run-experiment \
        --mode $mode \
        # set by . config-dataset-*.fish
        --dataset-config $dataset_config \
        (params --dataset-key $dataset_keys) \
        --dataset-split $dataset_split \
        # set by . config-model-*.fish
        --model-path $model_path \
        --model-tensor-parallel-size $model_tensor_parallel_size \
        --model-gpu-memory-utilization $model_gpu_memory_utilization \
        --model-parser $model_parser \
        --model-engine $model_engine \
        $model_quantization \
        $model_max_tokens \
        # prompt related
        --prompt-template $prompt_template \
        --prompt-system $prompt_system \
        --prompt-question $prompt_question \
        $argv
end
