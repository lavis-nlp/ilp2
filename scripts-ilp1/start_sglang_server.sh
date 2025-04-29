#!/bin/bash

# Überprüfen, ob Modellpfad übergeben wurde
if [ "$#" -ne 1 ]; then
	echo "Usage: $0 <model-path>"
	exit 1
fi

MODEL_PATH=$1

if ! [ -d "$MODEL_PATH" ]; then
	echo "Model path doesnt exist!"
	exit 1
fi


python -m sglang.launch_server --model-path "$MODEL_PATH" --port 30000 


