function run_experiments
    for n_cand in $n_cands
        for dataset in $datasets
            set output_prefix "$mode-$dataset-"

            if begin
                    test "$mode" = prompt-re-ranking
                    or test "$mode" = full-re-ranking
                    or test "$mode" = ranker-results
                end

                set output_prefix "$output_prefix$n_cand-"

                if test -n "$vertex_name"
                    set output_prefix "$output_prefixv-"
                end
            end

            if set -q output_suffix
                set output_prefix "$output_prefix$output_suffix-"
            end

            echo poetry run ilp run-experiment \
                --split $split \
                --model $model \
                --tensor-parallel-size 4 \
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
                --mode $mode $include_vertex_name \
                --output-prefix $output_prefix \
                $argv
        end
    end
end
