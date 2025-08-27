import requests
import json

# 本地部署的 API 地址
base_url = "http://192.168.1.5:7171/v1/chat/completions"
api_key = "sk-QD6JetjsxOxP38baqYZoQL3HTxbuRUAAozC68QM0Aw1JpOrN"
# 多模态 messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "请基于上面这张图片，简单描述一下画面内容。"},
            {"type": "image_url", "image_url": {"url": "https://www.baidu.com/img/PCtm_d9c8750bed0b3c7d089fa7d55720d6cf.png"}}
        ]
    }
]

# 如果要本地文件上传，可以使用 multipart/form-data，这里给出两种方式

## 方式一：直接在 JSON 中指定 image_url（如上）
payload = {
    "model": "gpt-4o",
    "messages": messages,

}
# 请求头，加入 Authorization
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
resp = requests.post(base_url,headers=headers, json=payload)
print("状态码：", resp.status_code)
print("回复：", json.dumps(resp.json(), ensure_ascii=False, indent=2))