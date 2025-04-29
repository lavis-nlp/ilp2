set model \
    --model /data/hiwi/lukas/llms/llama3/Meta-Llama-3-8B-Instruct \
    --tensor-parallel-size 1 \
    --sampling-use-beam-search y \
    --sampling-early-stopping y \
    --sampling-best-of 4
