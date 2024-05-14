#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <model-path> <tensor-parallel-size>"
    exit 1
fi

model-path=$1
tensor-parallel-size=$2

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_folder="Results-$timestamp"
mkdir "$results_folder"

datasets=("irt2-cde-tiny" "irt2-cde-small" "irt2-cde-medium" "irt2-cde-large")

for dataset in "${datasets[@]}"; do
    output_file="$results_folder/$dataset.txt"
    python_script.py "$dataset" "$model-path" "$tensor-parallel-size" | tee "$output_file"
done

