import ray
import requests
import torch
import wandb
import swanlab
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from src.reward.rewarder import Rewarder,ChunkRewarder
from utils.scp_tools import parallel_scp
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
BASE_URL = "http://192.168.1.6:7333"
# wandb.init(
#     project="reward-record",
#     name = "250801"
#
# )
swanlab.init(
    # 设置项目名
    project="reward-record",
    experiment_name = "250811"
)
settings = {
    "initial_steps": 200
}
# Configure Ray and GPU resources
def init_ray(model_path: str, num_gpus: Optional[int] = None):
    if not ray.is_initialized():
        # Auto-detect GPUs if not provided
        gpu_count = num_gpus or torch.cuda.device_count()
        ray.init(num_gpus=gpu_count)

    # Create one actor per GPU
    actors = []
    for i in range(torch.cuda.device_count()):
        actor = RewarderActor.options(num_gpus=1).remote(model_path)
        actors.append(actor)
    return actors

# Actor wrapper around Rewarder
@ray.remote(num_gpus=1)
class RewarderActor:
    def __init__(self, model_path: str):
        self.rewarder = ChunkRewarder(chunk_size=4200,model_path = model_path)

    def reward(self, response: str, image_path: str, visualize: bool = False, visual_save: Optional[str] = None):
        return self.rewarder.reward(response, image_path, visualize, visual_save)

# Pydantic models for requests and responses
class RewardRequest(BaseModel):
    response: str
    image_path: str
    visualize: Optional[bool] = False
    visual_save: Optional[str] = None

class RewardResult(BaseModel):
    shift_reward: float
    zoom_reward: float
    a_format_reward: float
    s_format_reward: float
    valid_action_reward: float

class BatchRequest(BaseModel):
    requests: List[RewardRequest]

class BatchResponse(BaseModel):
    results: List[RewardResult]

# Initialize FastAPI
app = FastAPI()
actors = []

def round_robin_dispatch(reqs, actors,new_image_paths):
    futures = []
    for idx, req in enumerate(reqs):
        actor = actors[idx % len(actors)]
        futures.append(
            actor.reward.remote(
                req.response,
                new_image_paths[idx],
                req.visualize,
                req.visual_save
            )
        )
        print("#"*15+f"RESPONSE{idx}:"+"#"*15)
        print(req.response)
    return futures

@app.on_event("startup")
def startup_event():
    global actors
    # Adjust the model path as needed
    model_path = "/data/wangzhenchuan/models/Qwen2___5-VL-7B-Instruct"
    init_ray(model_path)
    actors = init_ray(model_path)

@app.post("/rewards", response_model=BatchResponse)
def get_rewards(batch: BatchRequest):
    # 打印GPU占用

    # torch清除缓存
    torch.cuda.empty_cache()
    remote_paths = [req.image_path for req in batch.requests]
    responses = [req.response for req in batch.requests]
    # with open("./resp.txt", "w", encoding="utf-8") as f:
    #     f.write(str(responses))
    # print(responses)
    # request_response = requests.post(url=f"{BASE_URL}/get_valid_action_rewards", json={"responses":responses}).json()
    # valid_action_rewards = request_response['rewards']
    valid_action_rewards = [0.0 for _ in responses]

    new_image_paths = parallel_scp(remote_paths=remote_paths)
    global actors
    if not actors:
        raise HTTPException(status_code=500, detail="Actors not initialized")

    # Dispatch in round-robin fashion
    futures = round_robin_dispatch(batch.requests, actors,new_image_paths)
    try:
        raw_results = ray.get(futures)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # 将 raw_results 和 valid_action_rewards 合并，构造 Pydantic 返回模型
    results = []
    avg_sr = 0
    avg_zr = 0
    avg_af = 0
    avg_sf = 0
    avg_val = 0
    for idx, (sr, zr, af, sf) in enumerate(raw_results):
        # 如果 valid_action_rewards 长度不够，则默认 0.0
        val = valid_action_rewards[idx] if idx < len(valid_action_rewards) else 0.0
        results.append(
            RewardResult(
                shift_reward=sr,
                zoom_reward=zr,
                a_format_reward=af,
                s_format_reward=sf,
                valid_action_reward=val
            )
        )
        avg_val += val
        avg_sf += sf
        avg_zr += zr
        avg_af += af
        avg_sr += sr

    avg_val /= len(raw_results)
    avg_sf /= len(raw_results)
    avg_zr /= len(raw_results)
    avg_af /= len(raw_results)
    avg_sr /= len(raw_results)
    # wandb 记录所有类型的奖励
    swanlab.log({
        "steps": settings['initial_steps'],
        "shift_reward": avg_sr,
        "zoom_reward": avg_zr,
        "a_format_reward": avg_af,
        "s_format_reward": avg_sf,
        "valid_action_reward": avg_val
    })
    settings['initial_steps'] += 1
    return BatchResponse(results=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7452)
