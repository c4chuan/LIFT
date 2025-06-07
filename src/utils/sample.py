import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor,AutoTokenizer,AutoModelForCausalLM
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import dashscope
import torch

class ResponseSampler:
    def __init__(self, model_path):
        tokenizer, model, processor = self._load_model(model_path)
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor

    def sample(self,messages, n):
        """ sample函数输入messages，采样出n个回答 """
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        sampling_params = SamplingParams(
            n=n,
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=2048,
            stop_token_ids=[],
        )
        image_inputs, video_inputs,video_kwargs = process_vision_info(messages,return_video_kwargs=True)
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        # num_patches = torch.prod(image_inputs["image_grid_thw"]) / (2 ** 2) # mergesize的平方
        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            # FPS will be returned in video_kwargs
            "mm_processor_kwargs": video_kwargs,
        }
        outputs = self.model.generate([llm_inputs], sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text
        print(generated_text)
        return outputs

    def _load_model(self, model_path):
        model = LLM(
            model=model_path,
            limit_mm_per_prompt={"image": 2, "video": 0},
        )

        processor = AutoProcessor.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

        return tokenizer, model, processor