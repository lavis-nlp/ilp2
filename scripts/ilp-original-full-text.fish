#!/usr/bin/env fish

# model performance on all datasets (both all and subsampled)
# with the hyperparameters determined by the ilp-ablation-* scripts

set root (dirname (status -f))

set prompt_conf generic-with-text

echo
and . "$root"/ilp-config-model-llama3-70b.fish
and . "$root"/ilp-config-prompt-$prompt_conf.fish
and set prefix --output-prefix original-$prompt-conf-
or return


set ilp poetry run ilp run-experiment
for filename in original-subsampled # original
    set dataset_config --dataset-config lib/irt2/conf/datasets/$filename.yaml
    for dataset in tiny small medium large
        $ilp --split test $prompts $model $dataset_config \
            --datasets irt2/$dataset \
            --texts "lib/irt2/data/irt2/irt2-cde-"$dataset"/open.test-contexts-31189-30.pkl"
    end
end
