set model_parser csv
set model_engine vllm
set model_path /data/tyler/llms/DeepSeek-R1-Distill-Llama-70B
set model_dtype bfloat16


# vllm model params for 4 x RTX A6000
set model_tensor_parallel_size 4
set model_gpu_memory_utilization 0.8
set model_quantization --quantization fp8

# default params as defined by provider
set default_temperature --sampling-temperature 0.6
set default_top_p --sampling-top-p 0.95
