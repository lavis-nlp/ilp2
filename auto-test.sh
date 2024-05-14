#!/bin/bash

if [ $# -ne 4 ]; then
    echo "Usage: $0 <model_path> <tensor_parallel_size> <system_prompt_path> <question_template_path>"
    exit 1
fi

model_path=$1
tensor_parallel_size=$2
system_prompt_path=$3
question_template_path=$4

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_folder="Results-$timestamp"
mkdir "$results_folder"

datasets=("irt2-cde-tiny" "irt2-cde-small" "irt2-cde-medium" "irt2-cde-large")

for dataset in "${datasets[@]}"; do
    output_file="$results_folder/$dataset.txt"
    python run_test.py "$dataset" "$model_path" "$tensor_parallel_size" "$system_prompt_path" "$question_template_path" | tee "$output_file"
done
