export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_API_KEY=01f4ba887832dddfc01690cf5ac3fd9a9ad0a980
export HTTP_PROXY=http://wangzhenchuan:144987@81.70.105.191:7891
export HTTPS_PROXY=http://wangzhenchuan:144987@81.70.105.191:7891
cd /data/wangzhenchuan/Projects/LIFT/LLaMA-Factory
FORCE_TORCHRUN=1
llamafactory-cli train examples/train_full/qwen2_5vl_full_sft_LIFT_D.yaml