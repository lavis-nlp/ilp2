set parser csv
set engine vllm
set model /data/tyler/llms/DeepSeek-R1-Distill-Llama-70B
set dtype bfloat16
set rep_pen 1

# vllm model params for 4 x RTX A6000
set tensor_parallel_size 4
set gpu_memory_utilization 0.8
set quantization --quantization fp8
