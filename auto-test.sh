#!/bin/bash

if [ $# -ne 5 ]; then
    echo "Usage: $0 <model_path> <tensor_parallel_size> <system_prompt_path> <question_template_path>"
    exit 1
fi

model_name=$1
model_path=$2
tensor_parallel_size=$3
system_prompt_path=$4
question_template_path=$5

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_folder="results"
mkdir -p "$results_folder"

datasets=("irt2-cde-tiny" "irt2-cde-small" "irt2-cde-medium" "irt2-cde-large")

for dataset in "${datasets[@]}"; do
    result_dir="$results_folder/$model_name/$timestamp/$dataset/"
    mkdir -p "$result_dir"
    log_file="$result_dir/log.txt"
    error_log_file="$result_dir/errors.txt"
    python run_test.py "$dataset" "$model_path" "$tensor_parallel_size" "$system_prompt_path" "$question_template_path" "$result_dir" 2>$error_log_file | tee "$log_file"
done
