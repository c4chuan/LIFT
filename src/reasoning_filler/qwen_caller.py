"""
Qwen API 调用模块

负责调用 DashScope API 获取推理内容
"""

import time
import base64
from io import BytesIO
from typing import Dict, Any, List, Optional
from PIL import Image

try:
    import dashscope
    from dashscope import MultiModalConversation
except ImportError:
    print("警告: dashscope 未安装，请运行: pip install dashscope")
    dashscope = None


class QwenCaller:
    """Qwen API 调用器"""

    def __init__(
        self,
        api_key: str,
        model_name: str = "qwen-vl-plus",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: int = 60
    ):
        """
        初始化 Qwen API 调用器

        Args:
            api_key: DashScope API Key
            model_name: 模型名称，如 "qwen-vl-plus" 或 "qwen-vl-max"
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            timeout: 请求超时时间（秒）
        """
        if dashscope is None:
            raise ImportError("dashscope 未安装，请运行: pip install dashscope")

        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

        # 设置 API key
        dashscope.api_key = api_key

    def call(self, messages: List[Dict[str, Any]]) -> str:
        """
        调用 Qwen API

        Args:
            messages: 消息列表，符合 Qwen VL 格式

        Returns:
            模型生成的文本

        Raises:
            Exception: API 调用失败
        """
        # 转换消息格式为 DashScope 格式
        formatted_messages = self._format_messages(messages)

        for attempt in range(self.max_retries):
            try:
                response = MultiModalConversation.call(
                    model=self.model_name,
                    messages=formatted_messages,
                    timeout=self.timeout
                )

                # 检查响应状态
                if response.status_code == 200:
                    # 提取生成的文本
                    output = response.output
                    if output and "choices" in output:
                        return output["choices"][0]["message"]["content"][0]['text']
                    else:
                        raise Exception(f"响应格式异常: {response}")
                else:
                    error_msg = f"API 调用失败 (status: {response.status_code}): {response.message}"
                    if attempt < self.max_retries - 1:
                        print(f"{error_msg}，{self.retry_delay}秒后重试...")
                        time.sleep(self.retry_delay * (2 ** attempt))  # 指数退避
                        continue
                    else:
                        raise Exception(error_msg)

            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"调用出错: {e}，{self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                else:
                    raise Exception(f"API 调用失败（已重试{self.max_retries}次）: {e}")

        raise Exception("API 调用失败：超过最大重试次数")

    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将消息格式转换为 DashScope 格式

        Args:
            messages: 原始消息列表

        Returns:
            DashScope 格式的消息列表
        """
        formatted = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            # 如果 content 是字符串，直接使用
            if isinstance(content, str):
                formatted.append({
                    "role": role,
                    "content": [{"text": content}]
                })

            # 如果 content 是列表（包含文本和图片）
            elif isinstance(content, list):
                formatted_content = []

                for item in content:
                    item_type = item.get("type")

                    if item_type == "text":
                        formatted_content.append({"text": item.get("text", "")})

                    elif item_type == "image":
                        # 将 PIL Image 转换为 base64 或 URL
                        image = item.get("image")
                        if isinstance(image, Image.Image):
                            # 转换为 base64
                            image_url = self._pil_to_base64(image)
                            formatted_content.append({"image": image_url})
                        elif isinstance(image, str):
                            # 假设已经是 URL 或 base64
                            formatted_content.append({"image": image})

                formatted.append({
                    "role": role,
                    "content": formatted_content
                })

            else:
                # 其他情况，尝试转为字符串
                formatted.append({
                    "role": role,
                    "content": [{"text": str(content)}]
                })

        return formatted

    @staticmethod
    def _pil_to_base64(image: Image.Image) -> str:
        """
        将 PIL Image 转换为 base64 字符串

        Args:
            image: PIL Image 对象

        Returns:
            base64 格式的图片 URL
        """
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"


def main():
    """测试函数"""
    import os

    # 从环境变量获取 API key
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("请设置环境变量 DASHSCOPE_API_KEY")
        return

    caller = QwenCaller(api_key=api_key)

    # 测试简单文本调用
    print("=== 测试文本调用 ===")
    messages = [
        {
            "role": "system",
            "content": "你是一个有帮助的助手。"
        },
        {
            "role": "user",
            "content": "你好，请简单介绍一下你自己。"
        }
    ]

    try:
        response = caller.call(messages)
        print(f"响应: {response}")
    except Exception as e:
        print(f"调用失败: {e}")


if __name__ == "__main__":
    main()
