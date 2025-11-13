#!/usr/bin/env python3
"""
Reward Top Analyzer - 分析rollout数据中的最高/最低奖励样本

该工具从rollout数据目录中找出各种奖励指标的top5样本，并生成HTML可视化页面。
支持按训练步骤范围过滤数据。
"""

import json
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import html
import heapq


class RewardRecord:
    """奖励记录数据类"""
    def __init__(self, data: dict, step: int, record_idx: int):
        self.data = data
        self.step = step  # 训练步骤（来自文件名）
        self.record_idx = record_idx  # 文件内记录索引
        self.shift_reward = data.get('shift_reward', 0.0)
        self.zoom_reward = data.get('zoom_reward', 0.0)
        self.total_reward = self.shift_reward + self.zoom_reward
        self.score = data.get('score', 0.0)

    def __repr__(self):
        return f"Record(step={self.step}, idx={self.record_idx}, shift={self.shift_reward:.4f}, zoom={self.zoom_reward:.4f})"


def collect_records(input_dir: Path, start_step: int = None, end_step: int = None) -> List[RewardRecord]:
    """
    收集指定步骤范围内的所有rollout记录

    Args:
        input_dir: 输入目录路径
        start_step: 起始步骤（包含），None表示不限制
        end_step: 结束步骤（包含），None表示不限制

    Returns:
        所有符合条件的记录列表
    """
    records = []
    jsonl_files = sorted(input_dir.glob('*.jsonl'))

    print(f"找到 {len(jsonl_files)} 个JSONL文件")

    for jsonl_file in jsonl_files:
        # 从文件名提取步骤号
        try:
            step = int(jsonl_file.stem)
        except ValueError:
            print(f"警告: 跳过非数字命名的文件: {jsonl_file.name}")
            continue

        # 检查步骤范围
        if start_step is not None and step < start_step:
            continue
        if end_step is not None and step > end_step:
            continue

        # 读取文件中的所有记录
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                lines = content.split('\n')

                for idx, line in enumerate(lines):
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            record = RewardRecord(data, step, idx)
                            records.append(record)
                        except json.JSONDecodeError as e:
                            print(f"警告: 文件 {jsonl_file.name} 第 {idx+1} 行JSON解析失败: {e}")
                            continue
        except Exception as e:
            print(f"错误: 读取文件 {jsonl_file.name} 失败: {e}")
            continue

    print(f"收集到 {len(records)} 条记录（步骤范围: {start_step or '不限'} ~ {end_step or '不限'}）")
    return records


def find_top_samples(records: List[RewardRecord]) -> Dict[str, List[RewardRecord]]:
    """
    找出各项指标的top5样本

    Returns:
        包含5组top5样本的字典
    """
    results = {}

    # 1. zoom_reward 最高 top5
    results['zoom_highest'] = heapq.nlargest(5, records, key=lambda r: r.zoom_reward)

    # 2. zoom_reward 最低 top5（排除0）
    non_zero_zoom = [r for r in records if r.zoom_reward != 0.0]
    results['zoom_lowest'] = heapq.nsmallest(5, non_zero_zoom, key=lambda r: r.zoom_reward)

    # 3. shift_reward 最高 top5
    results['shift_highest'] = heapq.nlargest(5, records, key=lambda r: r.shift_reward)

    # 4. shift_reward 最低 top5
    results['shift_lowest'] = heapq.nsmallest(5, records, key=lambda r: r.shift_reward)

    # 5. shift_reward + zoom_reward 总和最高 top5
    results['total_highest'] = heapq.nlargest(5, records, key=lambda r: r.total_reward)

    # 打印统计信息
    print("\n=== Top5 样本统计 ===")
    for category, samples in results.items():
        print(f"\n{category}:")
        for i, record in enumerate(samples, 1):
            print(f"  {i}. Step {record.step}, Record {record.record_idx}: "
                  f"shift={record.shift_reward:.4f}, zoom={record.zoom_reward:.4f}, "
                  f"total={record.total_reward:.4f}")

    return results


def export_samples(top_samples: Dict[str, List[RewardRecord]], output_dir: Path):
    """
    导出top样本到分类目录

    Args:
        top_samples: top样本字典
        output_dir: 输出根目录
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    for category, samples in top_samples.items():
        category_dir = output_dir / category
        category_dir.mkdir(exist_ok=True)

        for i, record in enumerate(samples, 1):
            # 文件名包含排名、步骤和记录索引
            filename = f"rank{i}_step{record.step}_rec{record.record_idx}.jsonl"
            output_file = category_dir / filename

            # 写入JSON数据
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(record.data, f, ensure_ascii=False, indent=2)

    print(f"\n已导出所有样本到: {output_dir}")


def copy_original_htmls(top_samples: Dict[str, List[RewardRecord]],
                        html_dir: Path, output_dir: Path):
    """
    复制原始HTML文件到结果目录

    Args:
        top_samples: top样本字典
        html_dir: 原始HTML目录
        output_dir: 输出目录
    """
    import shutil

    if not html_dir or not html_dir.exists():
        print(f"\n警告: 原始HTML目录不存在: {html_dir}")
        print("跳过复制原始HTML文件")
        return set()

    # 创建存放原始HTML的目录
    original_html_dir = output_dir / 'original_steps'
    original_html_dir.mkdir(exist_ok=True)

    # 收集所有需要复制的步骤（去重）
    steps_to_copy = set()
    for samples in top_samples.values():
        for record in samples:
            steps_to_copy.add(record.step)

    print(f"\n复制原始HTML文件（共 {len(steps_to_copy)} 个步骤）...")

    copied_steps = set()
    for step in sorted(steps_to_copy):
        source_file = html_dir / f"{step}.html"
        if source_file.exists():
            dest_file = original_html_dir / f"{step}.html"
            shutil.copy2(source_file, dest_file)
            copied_steps.add(step)
            print(f"  复制: {step}.html")
        else:
            print(f"  警告: 未找到 {source_file.name}")

    print(f"✅ 已复制 {len(copied_steps)} 个原始HTML文件")
    return copied_steps


def create_html_template():
    """创建HTML模板（基于jsonl_to_html_converter.py）"""
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .category-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin: 10px;
            font-size: 14px;
        }}
        .zoom-highest {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }}
        .zoom-lowest {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }}
        .shift-highest {{ background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; }}
        .shift-lowest {{ background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; }}
        .total-highest {{ background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white; }}
        .section {{
            margin: 30px 0;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            overflow: hidden;
        }}
        .section-header {{
            background: #f8f9fa;
            padding: 15px;
            font-weight: bold;
            font-size: 18px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .section-content {{
            padding: 20px;
        }}
        .reward-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .reward-item {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ff9800;
            text-align: center;
        }}
        .reward-item h4 {{
            margin: 0 0 10px 0;
            color: #333;
            font-size: 14px;
        }}
        .reward-value {{
            font-size: 28px;
            font-weight: bold;
            color: #ff9800;
        }}
        .reward-item.highlight {{
            border-left: 4px solid #e91e63;
            background: linear-gradient(135deg, #fff5f7 0%, #ffe8ee 100%);
        }}
        .reward-item.highlight .reward-value {{
            color: #e91e63;
            font-size: 32px;
        }}
        .code-block {{
            background: #f5f5f5;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            white-space: pre-wrap;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 14px;
            border-left: 4px solid #2196f3;
            margin: 10px 0;
            max-height: 600px;
            overflow-y: auto;
        }}
        .image-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .image-container {{
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            overflow: hidden;
            transition: transform 0.2s;
            cursor: pointer;
        }}
        .image-container:hover {{
            transform: scale(1.02);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .image-container img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
        }}
        .modal-content {{
            margin: auto;
            display: block;
            max-width: 90%;
            max-height: 90%;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }}
        .close {{
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }}
        .navigation {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        .navigation a {{
            display: block;
            margin: 8px 0;
            color: #1976d2;
            text-decoration: none;
            font-weight: 500;
        }}
        .navigation a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        {content}
    </div>

    <div class="navigation">
        <h4 style="margin-top: 0;">导航</h4>
        <a href="index.html">← 返回索引</a>
    </div>

    <div id="imageModal" class="modal">
        <span class="close" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>

    <script>
        function showImage(imgSrc) {{
            document.getElementById("imageModal").style.display = "block";
            document.getElementById("modalImage").src = imgSrc;
        }}
        function closeModal() {{
            document.getElementById("imageModal").style.display = "none";
        }}
        window.onclick = function(event) {{
            if (event.target == document.getElementById("imageModal")) {{
                closeModal();
            }}
        }}
        document.addEventListener('keydown', function(event) {{
            if (event.key === "Escape") closeModal();
        }});
        document.addEventListener('DOMContentLoaded', function() {{
            var images = document.querySelectorAll('.image-container img');
            images.forEach(function(img) {{
                img.addEventListener('click', function() {{
                    showImage(this.src);
                }});
            }});
        }});
    </script>
</body>
</html>
"""


def create_sample_html(record: RewardRecord, rank: int, category: str,
                       category_name: str, output_file: Path, has_original_html: bool = False):
    """
    为单个样本创建HTML页面

    Args:
        record: 奖励记录
        rank: 排名
        category: 类别ID
        category_name: 类别显示名称
        output_file: 输出HTML文件路径
        has_original_html: 是否存在原始HTML文件
    """
    # 提取数据
    data = record.data
    input_content = html.escape(data.get('input', '无输入数据'))
    output_content = html.escape(data.get('output', '无输出数据'))

    # 创建图片部分
    images = data.get('images', [])
    if not isinstance(images, list):
        images = [images] if images else []

    image_items = []
    for idx, img_data in enumerate(images, 1):
        if isinstance(img_data, str):
            if img_data.startswith('data:image'):
                img_src = img_data
            elif img_data.startswith('iVBOR') or img_data.startswith('/9j/'):
                img_format = 'png' if img_data.startswith('iVBOR') else 'jpeg'
                img_src = f'data:image/{img_format};base64,{img_data}'
            else:
                img_src = f'data:image/png;base64,{img_data}'

            image_items.append(f'''
            <div class="image-container">
                <img src="{img_src}" alt="Screenshot {idx}">
            </div>
            ''')

    images_html = f'<div class="image-gallery">{"".join(image_items)}</div>' if image_items else '<p style="text-align:center;color:#999;">📷 无图片数据</p>'

    # 创建奖励展示（高亮相关指标）
    highlight_fields = {
        'zoom_highest': ['zoom_reward'],
        'zoom_lowest': ['zoom_reward'],
        'shift_highest': ['shift_reward'],
        'shift_lowest': ['shift_reward'],
        'total_highest': ['shift_reward', 'zoom_reward']
    }

    highlights = highlight_fields.get(category, [])

    reward_items = []
    rewards = [
        ('Shift Reward', record.shift_reward, 'shift_reward'),
        ('Zoom Reward', record.zoom_reward, 'zoom_reward'),
        ('Total Reward', record.total_reward, None),
        ('Score', record.score, None),
    ]

    for name, value, field in rewards:
        highlight_class = 'highlight' if field in highlights else ''
        reward_items.append(f'''
        <div class="reward-item {highlight_class}">
            <h4>{name}</h4>
            <div class="reward-value">{value:.6f}</div>
        </div>
        ''')

    # 原始HTML链接按钮
    original_html_button = ''
    if has_original_html:
        original_html_button = f'''
        <div style="margin-top: 20px;">
            <a href="../original_steps/{record.step}.html"
               style="display: inline-block; padding: 12px 24px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                      color: white; text-decoration: none; border-radius: 25px; font-weight: bold; box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                      transition: transform 0.2s;"
               onmouseover="this.style.transform='translateY(-2px)'"
               onmouseout="this.style.transform='translateY(0)'">
                📋 查看完整步骤HTML（包含所有{record.step}步的记录）
            </a>
        </div>
        '''

    # 组装页面内容
    content = f'''
    <div class="header">
        <h1>Top样本详情</h1>
        <span class="category-badge {category}">{category_name}</span>
        <div style="margin-top: 15px; font-size: 16px;">
            <strong>排名:</strong> #{rank} |
            <strong>训练步骤:</strong> {record.step} |
            <strong>记录索引:</strong> {record.record_idx}
        </div>
        {original_html_button}
    </div>

    <div class="section">
        <div class="section-header">🏆 奖励指标</div>
        <div class="section-content">
            <div class="reward-grid">
                {"".join(reward_items)}
            </div>
        </div>
    </div>

    <div class="section">
        <div class="section-header">🖼️ 截图</div>
        <div class="section-content">
            {images_html}
        </div>
    </div>

    <div class="section">
        <div class="section-header">📥 输入提示</div>
        <div class="section-content">
            <div class="code-block">{input_content}</div>
        </div>
    </div>

    <div class="section">
        <div class="section-header">📤 模型输出</div>
        <div class="section-content">
            <div class="code-block">{output_content}</div>
        </div>
    </div>
    '''

    # 生成完整HTML
    html_content = create_html_template().format(
        title=f"Rank {rank} - {category_name}",
        content=content
    )

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)


def create_index_html(top_samples: Dict[str, List[RewardRecord]],
                      output_dir: Path, start_step: int, end_step: int, copied_steps: set = None):
    """
    创建索引页面

    Args:
        top_samples: top样本字典
        output_dir: 输出目录
        start_step: 分析的起始步骤
        end_step: 分析的结束步骤
        copied_steps: 已复制的原始HTML步骤集合
    """
    if copied_steps is None:
        copied_steps = set()
    category_names = {
        'zoom_highest': 'Zoom Reward 最高 Top5',
        'zoom_lowest': 'Zoom Reward 最低 Top5（非0）',
        'shift_highest': 'Shift Reward 最高 Top5',
        'shift_lowest': 'Shift Reward 最低 Top5',
        'total_highest': 'Total Reward 最高 Top5'
    }

    sections_html = []

    for category, samples in top_samples.items():
        category_name = category_names.get(category, category)

        items_html = []
        for i, record in enumerate(samples, 1):
            filename = f"{category}/rank{i}_step{record.step}_rec{record.record_idx}.html"

            # 原始HTML链接
            original_html_link = ''
            if record.step in copied_steps:
                original_html_link = f'<a href="original_steps/{record.step}.html" style="color: #43a047; text-decoration: none; font-weight: 500;">查看 →</a>'
            else:
                original_html_link = '<span style="color: #999;">-</span>'

            items_html.append(f'''
            <tr>
                <td style="text-align: center; font-weight: bold;">#{i}</td>
                <td style="text-align: center;">{record.step}</td>
                <td style="text-align: center;">{record.record_idx}</td>
                <td style="text-align: center; color: #2196f3; font-weight: bold;">{record.shift_reward:.6f}</td>
                <td style="text-align: center; color: #9c27b0; font-weight: bold;">{record.zoom_reward:.6f}</td>
                <td style="text-align: center; color: #f44336; font-weight: bold;">{record.total_reward:.6f}</td>
                <td style="text-align: center;">
                    <a href="{filename}" style="color: #1976d2; text-decoration: none; font-weight: 500;">查看详情 →</a>
                </td>
                <td style="text-align: center;">
                    {original_html_link}
                </td>
            </tr>
            ''')

        sections_html.append(f'''
        <div class="category-section">
            <h2><span class="category-badge {category}">{category_name}</span></h2>
            <table class="samples-table">
                <thead>
                    <tr>
                        <th>排名</th>
                        <th>训练步骤</th>
                        <th>记录索引</th>
                        <th>Shift Reward</th>
                        <th>Zoom Reward</th>
                        <th>Total Reward</th>
                        <th>样本详情</th>
                        <th>原始HTML</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(items_html)}
                </tbody>
            </table>
        </div>
        ''')

    step_range = f"{start_step or '不限'} ~ {end_step or '不限'}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    index_html = f'''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reward Top样本分析 - 索引</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 36px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .stats {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            text-align: center;
        }}
        .category-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin: 5px;
            font-size: 16px;
        }}
        .zoom-highest {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }}
        .zoom-lowest {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }}
        .shift-highest {{ background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; }}
        .shift-lowest {{ background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; }}
        .total-highest {{ background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white; }}
        .category-section {{
            margin: 40px 0;
            background: #f8f9fa;
            padding: 30px;
            border-radius: 8px;
        }}
        .category-section h2 {{
            margin-top: 0;
            text-align: center;
        }}
        .samples-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
            border-radius: 8px;
            overflow: hidden;
        }}
        .samples-table th {{
            background: #333;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        .samples-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .samples-table tr:hover {{
            background: #f5f5f5;
        }}
        .samples-table a {{
            color: #1976d2;
            text-decoration: none;
            font-weight: 500;
        }}
        .samples-table a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏆 Reward Top样本分析</h1>
            <p style="font-size: 18px; color: #666;">训练数据奖励指标分析报告</p>
        </div>

        <div class="stats">
            <div style="font-size: 18px; margin-bottom: 10px;">
                <strong>分析步骤范围:</strong> {step_range}
            </div>
            <div style="font-size: 16px;">
                <strong>生成时间:</strong> {timestamp}
            </div>
        </div>

        {"".join(sections_html)}
    </div>
</body>
</html>
'''

    index_file = output_dir / 'index.html'
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(index_html)

    print(f"\n索引页面已创建: {index_file}")


def generate_visualizations(top_samples: Dict[str, List[RewardRecord]],
                           output_dir: Path, start_step: int, end_step: int, copied_steps: set = None):
    """
    生成所有HTML可视化页面

    Args:
        top_samples: top样本字典
        output_dir: 输出目录
        start_step: 起始步骤
        end_step: 结束步骤
        copied_steps: 已复制的原始HTML步骤集合
    """
    if copied_steps is None:
        copied_steps = set()
    category_names = {
        'zoom_highest': 'Zoom Reward 最高',
        'zoom_lowest': 'Zoom Reward 最低（非0）',
        'shift_highest': 'Shift Reward 最高',
        'shift_lowest': 'Shift Reward 最低',
        'total_highest': 'Total Reward 最高'
    }

    print("\n开始生成HTML可视化页面...")

    for category, samples in top_samples.items():
        category_dir = output_dir / category
        category_dir.mkdir(exist_ok=True)
        category_name = category_names.get(category, category)

        for i, record in enumerate(samples, 1):
            filename = f"rank{i}_step{record.step}_rec{record.record_idx}.html"
            output_file = category_dir / filename

            has_original_html = record.step in copied_steps
            create_sample_html(record, i, category, category_name, output_file, has_original_html)
            print(f"  生成: {category}/{filename}")

    # 创建索引页面
    create_index_html(top_samples, output_dir, start_step, end_step, copied_steps)

    print(f"\n✅ 所有HTML页面已生成完毕!")
    print(f"请打开 {output_dir / 'index.html'} 查看结果")


def main():
    parser = argparse.ArgumentParser(
        description='分析rollout数据中的最高/最低奖励样本并生成可视化报告',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析所有步骤的数据
  python reward_top_analyzer.py --input-dir ../sup_rollout_data_dir_1012 --output-dir ../top_rewards_analysis

  # 仅分析步骤100-200的数据
  python reward_top_analyzer.py --input-dir ../sup_rollout_data_dir_1012 --output-dir ../top_rewards_analysis --start-step 100 --end-step 200
        """
    )

    parser.add_argument('--input-dir',
                       default='../sup_rollout_data_dir_1012',
                       help='输入目录（包含JSONL文件）')
    parser.add_argument('--output-dir',
                       default='../top_rewards_analysis',
                       help='输出目录（HTML可视化结果）')
    parser.add_argument('--start-step',
                       type=int,
                       default=None,
                       help='起始训练步骤（包含），默认不限制')
    parser.add_argument('--end-step',
                       type=int,
                       default=None,
                       help='结束训练步骤（包含），默认不限制')
    parser.add_argument('--html-dir',
                       default='../sup_rollout_data_html_dir_1012',
                       help='原始HTML文件目录（可选）')

    args = parser.parse_args()

    # 路径处理
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    html_dir = Path(args.html_dir) if args.html_dir else None

    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return 1

    print("=" * 60)
    print("Reward Top Analyzer - 奖励样本分析工具")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"HTML目录: {html_dir if html_dir else '未指定'}")
    print(f"步骤范围: {args.start_step or '不限'} ~ {args.end_step or '不限'}")
    print("=" * 60)

    # 1. 收集数据
    print("\n[1/5] 收集数据...")
    records = collect_records(input_dir, args.start_step, args.end_step)

    if not records:
        print("错误: 未找到任何符合条件的记录")
        return 1

    # 2. 找出top样本
    print("\n[2/5] 分析top样本...")
    top_samples = find_top_samples(records)

    # 3. 导出数据
    print("\n[3/5] 导出样本数据...")
    export_samples(top_samples, output_dir)

    # 4. 复制原始HTML文件
    print("\n[4/5] 复制原始HTML文件...")
    copied_steps = copy_original_htmls(top_samples, html_dir, output_dir)

    # 5. 生成可视化
    print("\n[5/5] 生成HTML可视化...")
    generate_visualizations(top_samples, output_dir, args.start_step, args.end_step, copied_steps)

    print("\n" + "=" * 60)
    print("✅ 分析完成!")
    print(f"📂 结果目录: {output_dir.absolute()}")
    print(f"🌐 打开浏览器访问: {(output_dir / 'index.html').absolute()}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
