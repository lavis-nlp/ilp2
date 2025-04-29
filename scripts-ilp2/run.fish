function run_experiments
    for n_cand in $n_cands
        for dataset in $datasets
            set output_prefix "$mode-"(echo $dataset- | sed 's|/|_|')

            poetry run ilp $debug run-experiment \
                --split $split \
                --model $model \
                --tensor-parallel-size $tensor_parallel_size \
                --gpu-memory-utilization $gpu_memory_utilization \
                --prompt-template $prompt_template \
                --system-prompt $system_prompt \
                --question-template $question_template \
                --dataset-config $dataset_config \
                --datasets $dataset \
                --parser $parser \
                --sampling-repetition-penalty $rep_pen \
                --n-candidates $n_cand \
                --mentions-per-candidate $mentions_per_candidate \
                --engine $engine \
                --dtype $dtype \
                --mode $mode \
                --output-prefix $output_prefix \
                $argv
        end
    end
end
