#!/usr/bin/env python3
"""
将ShareGPT格式数据可视化为HTML文件

使用方法:
    python src/scripts/visualize_sharegpt.py \
        --input_json LLaMA-Factory/data/trajectory_finetune_data.json \
        --output_dir data/visualization_sharegpt \
        --base_image_dir LLaMA-Factory
"""

import os
import sys
import json
import argparse
import base64
import re
from pathlib import Path
from typing import List, Dict, Any


def image_to_base64(image_path: str) -> str:
    """
    将图像文件转换为base64编码

    Args:
        image_path: 图像文件路径

    Returns:
        base64编码的data URL
    """
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
            b64_data = base64.b64encode(image_data).decode('utf-8')
            return f"data:image/png;base64,{b64_data}"
    except Exception as e:
        print(f"警告: 无法读取图像 {image_path}: {e}")
        # 返回一个占位符
        return "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='300'%3E%3Crect fill='%23ddd' width='400' height='300'/%3E%3Ctext x='50%25' y='50%25' text-anchor='middle' fill='%23999'%3EImage not found%3C/text%3E%3C/svg%3E"


def parse_lift_tags(content: str) -> str:
    """
    解析并高亮LIFT标签

    Args:
        content: 原始文本内容

    Returns:
        HTML格式的内容
    """
    # 转义HTML特殊字符
    content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    # 恢复LIFT标签
    content = content.replace('&lt;zoom in&gt;', '<zoom_in_tag>').replace('&lt;/zoom in&gt;', '</zoom_in_tag>')
    content = content.replace('&lt;shift&gt;', '<shift_tag>').replace('&lt;/shift&gt;', '</shift_tag>')
    content = content.replace('&lt;summary&gt;', '<summary_tag>').replace('&lt;/summary&gt;', '</summary_tag>')
    content = content.replace('&lt;action&gt;', '<action_tag>').replace('&lt;/action&gt;', '</action_tag>')

    # 处理换行
    content = content.replace('\n', '<br>\n')

    # 高亮LIFT标签
    content = re.sub(
        r'<zoom_in_tag>(.*?)</zoom_in_tag>',
        r'<div class="zoom-in"><strong>🔍 &lt;zoom in&gt;</strong><br>\1</div>',
        content,
        flags=re.DOTALL
    )

    content = re.sub(
        r'<shift_tag>(.*?)</shift_tag>',
        r'<div class="shift"><strong>↔️ &lt;shift&gt;</strong><br>\1</div>',
        content,
        flags=re.DOTALL
    )

    content = re.sub(
        r'<summary_tag>(.*?)</summary_tag>',
        r'<div class="summary"><strong>📝 &lt;summary&gt;</strong><br>\1</div>',
        content,
        flags=re.DOTALL
    )

    content = re.sub(
        r'<action_tag>(.*?)</action_tag>',
        r'<div class="action"><strong>⚡ &lt;action&gt;</strong><br>\1</div>',
        content,
        flags=re.DOTALL
    )

    return content


def render_message(message: Dict[str, str], images: List[str], image_counter: Dict[str, int], base_image_dir: str) -> str:
    """
    渲染单条消息为HTML

    Args:
        message: 消息字典
        images: 图像路径列表
        image_counter: 图像计数器
        base_image_dir: 图像基准目录

    Returns:
        HTML字符串
    """
    role = message['role']
    content = message['content']

    role_icons = {
        'system': '🔧',
        'user': '👤',
        'assistant': '🤖'
    }

    role_names = {
        'system': 'System',
        'user': 'User',
        'assistant': 'Assistant'
    }

    # 步骤1: 用占位符标记<image>位置，避免被转义
    image_placeholders = []

    def mark_image(match):
        idx = image_counter['count']
        image_counter['count'] += 1
        placeholder = f"___IMAGE_PLACEHOLDER_{idx}___"

        if idx < len(images):
            # 构建完整的图像路径
            image_path = os.path.join(base_image_dir, images[idx])
            img_b64 = image_to_base64(image_path)
            img_html = f'<div class="image-container"><img src="{img_b64}" alt="Step {idx}"></div>'
            image_placeholders.append((placeholder, img_html))
        else:
            image_placeholders.append((placeholder, ''))

        return placeholder

    content = re.sub(r'<image>', mark_image, content)

    # 步骤2: HTML转义（保留LIFT标签的原始文本）
    content = content.replace('&', '&amp;')
    content = content.replace('<', '&lt;')
    content = content.replace('>', '&gt;')

    # 步骤3: 换行符转<br>
    content = content.replace('\n', '<br>\n')

    # 步骤4: 替换占位符为真实图像HTML
    for placeholder, img_html in image_placeholders:
        content = content.replace(placeholder, img_html)

    return f'''
    <div class="message {role}">
        <div class="role-badge">{role_icons.get(role, '💬')} {role_names.get(role, role.title())}</div>
        <div class="content">{content}</div>
    </div>
    '''


def generate_conversation_html(conversation: Dict[str, Any], conv_index: int, base_image_dir: str) -> str:
    """
    生成单个对话的HTML页面

    Args:
        conversation: 对话数据
        conv_index: 对话索引
        base_image_dir: 图像基准目录

    Returns:
        HTML字符串
    """
    messages = conversation.get('messages', [])
    images = conversation.get('images', [])

    # 提取task_id（从第一个图像路径中）
    task_id = 'unknown'
    if images:
        # 从 "data/classifieds_trajectory_images/11/intent_0.png" 提取 11
        match = re.search(r'_trajectory_images/(\d+)/', images[0])
        if match:
            task_id = match.group(1)

    image_counter = {'count': 0}

    messages_html = []
    for message in messages:
        msg_html = render_message(message, images, image_counter, base_image_dir)
        messages_html.append(msg_html)

    html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trajectory Conversation - Task {task_id}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 40px;
        }}

        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #667eea;
        }}

        .header h1 {{
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .metadata {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 30px;
            border-left: 5px solid #667eea;
        }}

        .metadata-item {{
            display: inline-block;
            margin-right: 20px;
            color: #666;
        }}

        .metadata-item strong {{
            color: #333;
        }}

        .message {{
            margin: 25px 0;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .message:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }}

        .system {{
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border-left: 5px solid #2196f3;
        }}

        .user {{
            background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
            border-left: 5px solid #ff9800;
        }}

        .assistant {{
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            border-left: 5px solid #4caf50;
        }}

        .role-badge {{
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 12px;
            color: #333;
        }}

        .content {{
            color: #444;
            line-height: 1.8;
            font-size: 15px;
        }}

        .image-container {{
            margin: 15px 0;
            text-align: center;
        }}

        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
            transition: transform 0.3s;
        }}

        .image-container img:hover {{
            transform: scale(1.02);
        }}

        /* LIFT标签样式 */
        .zoom-in {{
            background: #fff9c4;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #fbc02d;
            border-radius: 8px;
        }}

        .shift {{
            background: #f3e5f5;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #9c27b0;
            border-radius: 8px;
        }}

        .summary {{
            background: #e1f5fe;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #0288d1;
            border-radius: 8px;
        }}

        .action {{
            background: #ffebee;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #d32f2f;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
        }}

        .zoom-in strong, .shift strong, .summary strong, .action strong {{
            display: block;
            margin-bottom: 8px;
            font-size: 1.05em;
        }}

        .back-link {{
            display: inline-block;
            margin: 20px 0;
            padding: 12px 24px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 8px;
            transition: background 0.3s;
        }}

        .back-link:hover {{
            background: #5568d3;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 20px;
            }}

            .header h1 {{
                font-size: 1.8em;
            }}

            .message {{
                padding: 15px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <a href="index.html" class="back-link">← 返回索引</a>

        <div class="header">
            <h1>🎯 Trajectory Conversation</h1>
            <p style="color: #666; font-size: 1.1em;">Task #{task_id}</p>
        </div>

        <div class="metadata">
            <div class="metadata-item"><strong>📊 总消息数:</strong> {len(messages)}</div>
            <div class="metadata-item"><strong>🖼️ 图像数:</strong> {len(images)}</div>
            <div class="metadata-item"><strong>🔢 对话ID:</strong> {conv_index + 1}</div>
        </div>

        <div class="messages">
            {''.join(messages_html)}
        </div>

        <a href="index.html" class="back-link">← 返回索引</a>
    </div>
</body>
</html>'''

    return html_content


def generate_index_html(conversations: List[Dict[str, Any]]) -> str:
    """
    生成索引页面HTML

    Args:
        conversations: 所有对话数据

    Returns:
        HTML字符串
    """
    rows = []
    for idx, conv in enumerate(conversations):
        messages = conv.get('messages', [])
        images = conv.get('images', [])

        # 提取task_id和environment
        task_id = 'unknown'
        environment = 'unknown'
        if images:
            # 从 "data/classifieds_trajectory_images/11/intent_0.png" 提取 environment 和 task_id
            match = re.search(r'(\w+)_trajectory_images/(\d+)/', images[0])
            if match:
                environment = match.group(1)
                task_id = match.group(2)

        rows.append(f'''
        <tr>
            <td>{task_id}</td>
            <td><span class="env-badge">{environment}</span></td>
            <td>{len(messages)}</td>
            <td>{len(images)}</td>
            <td><a href="conversation_task_{task_id}.html" class="view-link">查看详情 →</a></td>
        </tr>
        ''')

    html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ShareGPT Trajectory Visualization Index</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 40px;
        }}

        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 30px;
            border-bottom: 3px solid #667eea;
        }}

        .header h1 {{
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .header p {{
            color: #666;
            font-size: 1.1em;
        }}

        .stats {{
            display: flex;
            justify-content: space-around;
            margin-bottom: 40px;
            gap: 20px;
        }}

        .stat-card {{
            flex: 1;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }}

        .stat-card h3 {{
            font-size: 2.5em;
            margin-bottom: 5px;
        }}

        .stat-card p {{
            font-size: 1em;
            opacity: 0.9;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }}

        thead {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}

        th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
            font-size: 1.05em;
        }}

        td {{
            padding: 15px;
            border-bottom: 1px solid #eee;
        }}

        tbody tr {{
            transition: background 0.2s;
        }}

        tbody tr:hover {{
            background: #f8f9fa;
        }}

        .env-badge {{
            display: inline-block;
            padding: 5px 12px;
            background: #e3f2fd;
            color: #1976d2;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 500;
        }}

        .view-link {{
            color: #667eea;
            text-decoration: none;
            font-weight: 500;
            transition: color 0.2s;
        }}

        .view-link:hover {{
            color: #5568d3;
        }}

        @media (max-width: 768px) {{
            .stats {{
                flex-direction: column;
            }}

            .header h1 {{
                font-size: 1.8em;
            }}

            table {{
                font-size: 0.9em;
            }}

            th, td {{
                padding: 10px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 ShareGPT Trajectory Visualization</h1>
            <p>Browser-based visualization of trajectory conversations</p>
        </div>

        <div class="stats">
            <div class="stat-card">
                <h3>{len(conversations)}</h3>
                <p>Total Conversations</p>
            </div>
            <div class="stat-card">
                <h3>{sum(len(c.get('messages', [])) for c in conversations)}</h3>
                <p>Total Messages</p>
            </div>
            <div class="stat-card">
                <h3>{sum(len(c.get('images', [])) for c in conversations)}</h3>
                <p>Total Images</p>
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Task ID</th>
                    <th>Environment</th>
                    <th>Messages</th>
                    <th>Images</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
</body>
</html>'''

    return html_content


def visualize_sharegpt(input_json: str, output_dir: str, base_image_dir: str) -> None:
    """
    将ShareGPT数据可视化为HTML

    Args:
        input_json: 输入JSON文件路径
        output_dir: 输出目录
        base_image_dir: 图像基准目录
    """
    print("="*60)
    print("ShareGPT数据HTML可视化")
    print("="*60)
    print(f"输入文件: {input_json}")
    print(f"输出目录: {output_dir}")
    print(f"图像基准: {base_image_dir}")
    print("="*60)

    # 读取JSON数据
    print("\n读取JSON数据...")
    with open(input_json, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    print(f"找到 {len(conversations)} 个对话")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成每个对话的HTML
    print("\n生成对话HTML文件...")
    for idx, conv in enumerate(conversations):
        # 提取task_id
        task_id = 'unknown'
        images = conv.get('images', [])
        if images:
            for image in images:
                match = re.search(r'_trajectory_images/(\d+)/', image)
                if match:
                    task_id = match.group(1)
                    break

        html_content = generate_conversation_html(conv, idx, base_image_dir)
        output_path = os.path.join(output_dir, f'conversation_task_{task_id}.html')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✓ 生成 conversation_task_{task_id}.html")

    # 生成索引页面
    print("\n生成索引页面...")
    index_html = generate_index_html(conversations)
    index_path = os.path.join(output_dir, 'index.html')

    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_html)

    print(f"✓ 生成 index.html")

    print("\n" + "="*60)
    print("可视化完成!")
    print("="*60)
    print(f"生成HTML文件数: {len(conversations) + 1}")
    print(f"索引页面: {index_path}")
    print(f"在浏览器中打开 {index_path} 查看")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="将ShareGPT格式数据可视化为HTML文件"
    )

    parser.add_argument(
        '--input_json',
        type=str,
        default='LLaMA-Factory/data/trajectory_finetune_data_d.json',
        help='ShareGPT格式的JSON输入文件路径'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/visualization_sharegpt_d',
        help='HTML输出目录'
    )

    parser.add_argument(
        '--base_image_dir',
        type=str,
        default='LLaMA-Factory',
        help='图像路径的基准目录'
    )

    args = parser.parse_args()

    # 执行可视化
    visualize_sharegpt(
        input_json=args.input_json,
        output_dir=args.output_dir,
        base_image_dir=args.base_image_dir
    )


if __name__ == '__main__':
    main()
