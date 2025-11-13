import json
import os

import requests

from src.utils.data_tools import dataset_construct
from src.prompts.prompt_construct import construct_messages_by_elements
from openai import OpenAI
import base64
import os

# 设置 OpenAI client
# client = OpenAI(
#     api_key="sk-QD6JetjsxOxP38baqYZoQL3HTxbuRUAAozC68QM0Aw1JpOrN",
#     base_url="https://xiaoai.plus/v1",
# )

client = OpenAI(
    api_key="sk-QD6JetjsxOxP38baqYZoQL3HTxbuRUAAozC68QM0Aw1JpOrN",
    base_url="http://192.168.1.5:7171/v1",
)
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
            max_tokens=2048,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"



def find_user_intent(task_id):
    """搜单个任务的intent"""
    data_list = dataset_construct()
    return data_list.select([task_id])['intent'][0]

def save_messages(messages, output_path):
    """保存messages"""
    with open(output_path, 'w') as f:
        # 保存为json
        json.dump(messages, f)

def main(inputs):
    """
    目标是输入task_id, environment截图，previous_actions，构建出可以用于测试的valid_messages
    :param task_id: task_id是为了找到User Intent
    :param environment_image_path: 当前环境的截图
    :param previous_actions: 之前的动作，文本格式
    :param output_path: 输出的目录，应该是直接到json
    :return: None
    """
    if os.path.exists(inputs['output_path']):
        print(f"task{inputs['task_id']}的valid messages已存在，不再进行生成")
        return

    task_id = inputs['task_id']
    url = inputs['url']
    environment_image_path = inputs['environment_image_path']
    previous_actions = inputs['previous_actions']
    output_path = inputs['output_path']
    # 找到task_id对应的user intent
    user_intent = find_user_intent(task_id)

    # 构建messages
    messages = construct_messages_by_elements(
        user_intent,
        url,
        environment_image_path,
        previous_actions,
    )

    # 保存
    save_messages(messages, output_path)

if __name__ == '__main__':

    # 下面这个是ibr的config
    # inputs = {
    #     'task_id': 58,
    #     'url': 'http://127.0.0.1:9980/index.php?page=search&sCategory=17',
    #     'environment_image_path': '/data/wangzhenchuan/Projects/LIFT/data/validation/ibr_valid_env.png',
    #     'previous_actions': "click on furniture category",
    #     'output_path': '/data/wangzhenchuan/Projects/LIFT/data/validation/ibr_valid_messages.json'
    # }

    inputs = {
        'task_id': 34,
        'url': 'http://127.0.0.1:9980/index.php?page=search&sPattern=paintings',
        'environment_image_path': '/data/wangzhenchuan/Projects/LIFT/data/validation/idr_valid_env.png',
        'previous_actions': "search for paintings",
        'output_path': '/data/wangzhenchuan/Projects/LIFT/data/validation/idr_valid_messages.json'
    }

    # main(
    #     inputs
    # )


    with open(inputs['output_path'], 'r') as f:
        messages = json.load(f)
        # response = requests.post(
        #     "http://127.0.0.1:7451/sample",
        #     json={
        #         "messages": messages,
        #         "n":1
        #     }
        # ).json()['samples'][0]
        t_messages = messages_format_transform(messages)
        response = call_gpt_4o(t_messages)
        print(response)





