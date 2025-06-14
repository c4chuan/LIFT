# persistent_api.py
import os
import json
from typing import List, Optional
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM,SamplingParams
from qwen_vl_utils import process_vision_info
from contextlib import asynccontextmanager

# ------------------------
# 定义请求/响应的数据模型
# ------------------------

class SampleRequest(BaseModel):
    messages: List
    n: Optional[int] = 1

class SampleResponse(BaseModel):
    samples: List[str]

# ------------------------
# 全局占位：将模型加载移至 lifespan 管理
# ------------------------
MODEL_PATH = os.getenv(
    "MODEL_PATH", "/data/wangzhenchuan/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct"
)
processor = None
tokenizer = None
model = None
sampler = None

# ------------------------
# 构造采样器类（延迟实例化）
# ------------------------
class ResponseSampler:
    def __init__(self, model, tokenizer, processor):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor

    def sample(self, messages: List, n: int) -> List[str]:
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        sampling_params = SamplingParams(
            n=n,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            max_tokens=2048,
            stop_token_ids=[],
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }

        outputs = self.model.generate([llm_inputs], sampling_params=sampling_params)
        return [out.text for out in outputs[0].outputs]

# ------------------------
# 定义 FastAPI 应用及 lifespan 事件
# ------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型
    global processor, tokenizer, model, sampler
    print(f"Loading model from {MODEL_PATH} ...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    model = LLM(model=MODEL_PATH,
                # enforce_eager=True,
                limit_mm_per_prompt={"image": 2, "video": 0})
    sampler = ResponseSampler(model, tokenizer, processor)
    print("Model loaded ✅")
    yield
    # 应用关闭时可做清理（可选）

app = FastAPI(
    title="Qwen2.5-VL Sampling API",
    lifespan=lifespan
)

@app.post("/sample", response_model=SampleResponse)
def sample_endpoint(req: SampleRequest):
    try:
        texts = sampler.sample(req.messages, req.n)
        return SampleResponse(samples=texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 启动时请确保使用模块名 persistent_api:app
    uvicorn.run("vllm_backend:app", host="0.0.0.0", port=7451, log_level="info")
