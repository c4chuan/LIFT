from openai import OpenAI
import requests
import time
import json
from reward.reward_tools import action_format_reward
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://192.168.1.6:7333"  # environment manager

# 设置 OpenAI client
client = OpenAI(
    api_key="sk-QD6JetjsxOxP38baqYZoQL3HTxbuRUAAozC68QM0Aw1JpOrN",
    base_url="https://xiaoai.plus/v1",
)

import base64
import os


def messages_format_transform(qwen_msgs):
    """
    把 qwen_2.5-vl 格式的消息列表，转成 OpenAI GPT-4o 接口可用的格式。

    Args:
        qwen_msgs (list of dict): 原始消息，每个 dict 包含 'role' 和 'content'，
            其中 content 是一个列表，每项可能有 'text' 和/或 'image'（本地路径）。
    Returns:
        list of dict: OpenAI ChatCompletion 调用所需的 messages 格式。
    """
    transformed = []

    for msg in qwen_msgs:
        blocks = []
        for part in msg.get("content", []):
            # 文本块
            if "text" in part:
                blocks.append({
                    "type": "text",
                    "text": part["text"]
                })
            # 图像块：读文件、Base64 编码、拼 data URI
            if "image" in part:
                img_path = part["image"]
                if os.path.isfile(img_path):
                    with open(img_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    # GPT-4o 要求 inline Base64 格式：data:image/png;base64,...
                    data_uri = f"data:image/png;base64,{b64}"
                    blocks.append({
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri,
                            "detail": "high"
                        }
                    })
                else:
                    raise FileNotFoundError(f"Image not found: {img_path}")

        # 如果只有单一文本块，可直接用字符串，否则用 list 块结构
        if len(blocks) == 1 and blocks[0]["type"] == "text":
            content = blocks[0]["text"]
        else:
            content = blocks

        transformed.append({
            "role": msg["role"],
            "content": content
        })

    return transformed

def call_gpt_4o(messages):
    """输入多模态 messages，调用 gpt-4o 得到回答"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1024,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def parallel_call_gpt_4o(messages_list, max_workers=4):
    """输入多模态 messages 列表，并行调用 gpt-4o 得到回复，结果顺序与输入一致"""
    responses = [None] * len(messages_list)
    # 使用线程池并行调用
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(call_gpt_4o, msgs): idx
            for idx, msgs in enumerate(messages_list)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                responses[idx] = future.result()
            except Exception as e:
                responses[idx] = f"Error: {str(e)}"
    return responses


def get_messages(num):
    """输入要取得的消息数目，取得一个 messages 的 list"""
    return requests.get(
        f"{BASE_URL}/get_messages", params={"num": num}
    ).json().get("messages", [])


def feed_responses(responses):
    """输入多模态 responses，将 responses 发送给 environment manager"""
    requests.post(
        f"{BASE_URL}/feed_responses", json=responses
    )


def check_responses(responses):
    """检查 responses 是否为合法动作，不合法时返回 '```None```'"""
    new_responses = []
    for response in responses:
        if action_format_reward(response) > 0:
            new_responses.append(response)
        else:
            new_responses.append("```None```")
    return new_responses


if __name__ == '__main__':
    num_messages = 4
    messages_to_save = []
    total_steps = 200 // num_messages
    cur_step = 0

    st_time = time.time()
    while cur_step < total_steps:
        # 获取环境消息
        messages = get_messages(num_messages)

        # 若 messages 是列表的列表，则直接并行；否则按单条消息列表化
        if messages and isinstance(messages[0], list):
            batch = messages
        else:
            batch = [[m] for m in messages]

        # 格式转换
        batch = [messages_format_transform(batch_msgs) for batch_msgs in batch]

        # 并行获取回复，保持顺序
        responses = parallel_call_gpt_4o(batch)
        print(responses[0])
        new_responses = check_responses(responses)

        # 发送给环境
        feed_responses(new_responses)

        # 保存对话
        for new_response,message in zip(new_responses, messages):
            message.append({
                "role": "assistant",
                "content": [
                    {
                        "text": new_response
                    }
                ]
            })
        messages_to_save+= messages
        with open("../data/LIFT_sft.json", "w", encoding="utf-8") as f:
            json.dump(messages_to_save, f, ensure_ascii=False, indent=2)

        cur_step += 1
        time.sleep(20)

    end_time = time.time()
    print(f"总共用时：{(end_time - st_time)/60:.2f} min")
    print(json.dumps(messages_to_save[-1], ensure_ascii=False, indent=2))

