#!/usr/bin/env bash
# 1) export visible GPUs
export CUDA_VISIBLE_DEVICES=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 2) initialize conda for this shell
eval "$(conda shell.bash hook)"

# 3) activate your environment
conda activate vllm-serve

# 4) launch vLLM server, with the model path as a positional argument
vllm serve /data/pretrained_models/Qwen2.5-VL-7B-Instruct \
    --served-model-name gpt-4o \
    --host 0.0.0.0 \
    --port 7171 \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --api-key sk-QD6JetjsxOxP38baqYZoQL3HTxbuRUAAozC68QM0Aw1JpOrN \
    --limit-mm-per-prompt image=17,video=0

#vllm serve /data/pretrained_models/Qwen2.5-VL-7B-Instruct \
#vllm serve /data/wangzhenchuan/Projects/LIFT/merged_models/qwen25_vl_7b_instruct_200steps \