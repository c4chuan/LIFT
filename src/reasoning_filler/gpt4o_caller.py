"""
GPT-4o API 调用模块

负责调用 OpenAI API (GPT-4o with vision) 获取推理内容
"""

import time
import base64
from io import BytesIO
from typing import Dict, Any, List
from PIL import Image

try:
    from openai import OpenAI
except ImportError:
    print("警告: openai 未安装，请运行: pip install openai")
    OpenAI = None


class GPT4oCaller:
    """GPT-4o API 调用器"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str = "gpt-4o",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: int = 60
    ):
        """
        初始化 GPT-4o API 调用器

        Args:
            api_key: OpenAI API Key
            base_url: OpenAI API Base URL
            model_name: 模型名称，如 "gpt-4o"
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            timeout: 请求超时时间（秒）
        """
        if OpenAI is None:
            raise ImportError("openai 未安装，请运行: pip install openai")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout
        )
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def call(self, messages: List[Dict[str, Any]]) -> str:
        """
        调用 GPT-4o API

        Args:
            messages: 消息列表，符合 OpenAI 格式

        Returns:
            模型生成的文本

        Raises:
            Exception: API 调用失败
        """
        # 转换消息格式（处理 PIL Image）
        formatted_messages = self._format_messages(messages)

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=formatted_messages,
                    temperature=0.7,
                    max_tokens=4096
                )

                # 提取生成的文本
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                    if content:
                        return content
                    else:
                        raise Exception("响应内容为空")
                else:
                    raise Exception(f"响应格式异常: {response}")

            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"GPT-4o 调用出错: {e}，{self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay * (2 ** attempt))  # 指数退避
                    continue
                else:
                    raise Exception(f"GPT-4o API 调用失败（已重试{self.max_retries}次）: {e}")

        raise Exception("GPT-4o API 调用失败：超过最大重试次数")

    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将消息格式转换为 OpenAI 格式（处理 PIL Image）

        Args:
            messages: 原始消息列表

        Returns:
            OpenAI 格式的消息列表
        """
        formatted = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            # 如果 content 是字符串，直接使用
            if isinstance(content, str):
                formatted.append({
                    "role": role,
                    "content": content
                })

            # 如果 content 是列表（包含文本和图片）
            elif isinstance(content, list):
                formatted_content = []

                for item in content:
                    item_type = item.get("type")

                    if item_type == "text":
                        formatted_content.append({
                            "type": "text",
                            "text": item.get("text", "")
                        })

                    elif item_type == "image":
                        # 处理 PIL Image 对象
                        image = item.get("image")
                        if isinstance(image, Image.Image):
                            # 转换为 base64 PNG (无损、原始分辨率)
                            image_b64 = self._pil_to_base64_png(image)
                            formatted_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                }
                            })
                        elif isinstance(image, str):
                            # 假设已经是 data URL
                            formatted_content.append({
                                "type": "image_url",
                                "image_url": {"url": image}
                            })

                    elif item_type == "image_url":
                        # 已经是 OpenAI 格式的 image_url
                        formatted_content.append(item)

                formatted.append({
                    "role": role,
                    "content": formatted_content
                })

            else:
                # 其他情况，尝试转为字符串
                formatted.append({
                    "role": role,
                    "content": str(content)
                })

        return formatted

    @staticmethod
    def _pil_to_base64_png(image: Image.Image) -> str:
        """
        将 PIL Image 转换为 base64 PNG 字符串（无损、原始分辨率）

        Args:
            image: PIL Image 对象

        Returns:
            base64 编码的 PNG 字符串
        """
        # Convert to RGB if RGBA (for PNG compatibility)
        if image.mode == 'RGBA':
            # Create white background for transparency
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
            image = background
        elif image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')

        # Save to bytes as PNG (lossless, no resize)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


def main():
    """测试函数"""
    import os

    # 从环境变量或配置获取 API 信息
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not api_key:
        print("请设置环境变量 OPENAI_API_KEY")
        return

    caller = GPT4oCaller(api_key=api_key, base_url=base_url)

    # 测试简单文本调用
    print("=== 测试 GPT-4o 文本调用 ===")
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Hello! Please introduce yourself briefly."
        }
    ]

    try:
        response = caller.call(messages)
        print(f"响应: {response}")
    except Exception as e:
        print(f"调用失败: {e}")


if __name__ == "__main__":
    main()
