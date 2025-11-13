#!/usr/bin/env python3
"""
JSONL to HTML Converter for Rollout Data

This script converts jsonl files from rollout_data_dir to HTML files 
for easy visualization of each step's input, output, and reward data.
"""

import json
import os
import argparse
from pathlib import Path
from datetime import datetime
import html

def create_html_template():
    """Create the basic HTML template"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rollout Data Step {step_num}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
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
        .input-section .section-header {{
            background: #e3f2fd;
            color: #1976d2;
        }}
        .output-section .section-header {{
            background: #e8f5e8;
            color: #388e3c;
        }}
        .reward-section .section-header {{
            background: #fff3e0;
            color: #f57c00;
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
        }}
        .reward-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .reward-item {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 4px;
            border-left: 4px solid #ff9800;
        }}
        .reward-item h4 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .reward-value {{
            font-size: 24px;
            font-weight: bold;
            color: #ff9800;
        }}
        .navigation {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 10px;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .navigation a {{
            display: block;
            margin: 5px 0;
            color: #1976d2;
            text-decoration: none;
        }}
        .navigation a:hover {{
            text-decoration: underline;
        }}
        .step-info {{
            background: #e3f2fd;
            padding: 10px 15px;
            border-radius: 4px;
            margin-bottom: 20px;
            border-left: 4px solid #1976d2;
        }}
        .record-separator {{
            background: #f8f9fa;
            padding: 15px;
            margin: 30px 0 20px 0;
            border-radius: 6px;
            border-left: 4px solid #2196f3;
            text-align: center;
        }}
        .record-separator h2 {{
            margin: 0 0 10px 0;
            color: #1976d2;
            font-size: 20px;
        }}
        .record-info {{
            color: #666;
            font-size: 14px;
        }}
        .record-divider {{
            margin: 40px 0;
            border: none;
            border-top: 2px solid #e0e0e0;
            background: linear-gradient(90deg, transparent, #e0e0e0, transparent);
        }}
        .image-section .section-header {{
            background: #f3e5f5;
            color: #7b1fa2;
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
        .image-caption {{
            background: #f8f9fa;
            padding: 8px;
            text-align: center;
            font-size: 12px;
            color: #666;
        }}
        /* 图片点击放大功能 */
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
        .close:hover {{
            color: #bbb;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Rollout Data Visualization</h1>
            <div class="step-info">
                <strong>Step:</strong> {step_num} | 
                <strong>Score:</strong> {score} | 
                <strong>Generated:</strong> {timestamp}
            </div>
        </div>

        {content}
    </div>

    <div class="navigation">
        <h4>Navigation</h4>
        <a href="index.html">← Back to Index</a>
    </div>

    <!-- 图片放大模态框 -->
    <div id="imageModal" class="modal">
        <span class="close" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>

    <script>
        // 图片点击放大功能
        function showImage(imgSrc) {{
            var modal = document.getElementById("imageModal");
            var modalImg = document.getElementById("modalImage");
            modal.style.display = "block";
            modalImg.src = imgSrc;
        }}

        function closeModal() {{
            document.getElementById("imageModal").style.display = "none";
        }}

        // 点击模态框外部关闭
        window.onclick = function(event) {{
            var modal = document.getElementById("imageModal");
            if (event.target == modal) {{
                closeModal();
            }}
        }}

        // ESC 键关闭
        document.addEventListener('keydown', function(event) {{
            if (event.key === "Escape") {{
                closeModal();
            }}
        }});

        // 为所有图片添加点击事件
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

def format_input_content(input_text):
    """Format the input content for better readability"""
    # Escape HTML and preserve formatting
    formatted = html.escape(input_text)
    
    # Add some basic formatting for common patterns
    lines = formatted.split('\n')
    formatted_lines = []
    
    for line in lines:
        # Highlight system/user/assistant markers
        if line.strip().startswith('system'):
            line = f'<strong style="color: #d32f2f;">{line}</strong>'
        elif line.strip().startswith('user'):
            line = f'<strong style="color: #1976d2;">{line}</strong>'
        elif line.strip().startswith('assistant'):
            line = f'<strong style="color: #388e3c;">{line}</strong>'
        
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def create_image_section(data, record_num):
    """Create the image section HTML from base64 encoded images"""
    images = data.get('images', [])

    if not images:
        return '''
        <div style="padding: 20px; text-align: center; color: #999;">
            📷 No images available for this record
        </div>
        '''

    # Handle both list and single image cases
    if not isinstance(images, list):
        images = [images]

    image_items = []
    for idx, img_data in enumerate(images, 1):
        # 检查是否已经包含 data URI scheme
        if isinstance(img_data, str):
            if img_data.startswith('data:image'):
                img_src = img_data
            elif img_data.startswith('iVBOR') or img_data.startswith('/9j/'):
                # PNG or JPEG base64 without data URI
                img_format = 'png' if img_data.startswith('iVBOR') else 'jpeg'
                img_src = f'data:image/{img_format};base64,{img_data}'
            else:
                # 假设是其他格式的 base64
                img_src = f'data:image/png;base64,{img_data}'
        else:
            continue

        image_items.append(f'''
        <div class="image-container">
            <img src="{img_src}" alt="Screenshot {idx}">
            <div class="image-caption">Screenshot {idx}</div>
        </div>
        ''')

    if not image_items:
        return '''
        <div style="padding: 20px; text-align: center; color: #999;">
            📷 No valid images available for this record
        </div>
        '''

    return f'''
    <div class="image-gallery">
        {''.join(image_items)}
    </div>
    '''

def create_reward_section(data):
    """Create the reward section HTML"""
    reward_items = []

    # Main score
    if 'score' in data:
        reward_items.append(f'''
        <div class="reward-item">
            <h4>Total Score</h4>
            <div class="reward-value">{data['score']:.4f}</div>
        </div>
        ''')

    # Individual reward components
    reward_fields = ['shift_reward', 'zoom_reward', 'format_reward', 'valid_action_reward']
    for field in reward_fields:
        if field in data:
            reward_items.append(f'''
            <div class="reward-item">
                <h4>{field.replace('_', ' ').title()}</h4>
                <div class="reward-value">{data[field]:.4f}</div>
            </div>
            ''')

    return f'''
    <div class="reward-grid">
        {''.join(reward_items)}
    </div>
    '''

def convert_jsonl_to_html(jsonl_file, output_dir):
    """Convert a single JSONL file to HTML"""
    step_num = Path(jsonl_file).stem
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Try to parse as multiple JSON objects (true JSONL format)
        lines = content.split('\n')
        data_records = []
        
        for line in lines:
            line = line.strip()
            if line:
                try:
                    data_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        # If no data found, try parsing as single JSON object
        if not data_records:
            try:
                data_records = [json.loads(content)]
            except json.JSONDecodeError:
                print(f"Error: No valid JSON found in {jsonl_file}")
                return None
                
    except Exception as e:
        print(f"Error reading {jsonl_file}: {e}")
        return None
    
    # Create content sections for all data records
    content_sections = []
    total_records = len(data_records)
    
    for i, data in enumerate(data_records, 1):
        # Extract data
        input_content = data.get('input', 'No input data available')
        output_content = data.get('output', 'No output data available')
        score = data.get('score', 0.0)
        
        # Record header
        record_header = f'''
        <div class="record-separator">
            <h2>📊 Record {i} of {total_records}</h2>
            <div class="record-info">Score: <strong>{score:.4f}</strong></div>
        </div>
        '''
        content_sections.append(record_header)
        
        # Image section - 添加在最前面
        image_html = create_image_section(data, i)
        content_sections.append(f'''
        <div class="section image-section">
            <div class="section-header">🖼️ Screenshots (Record {i})</div>
            <div class="section-content">
                {image_html}
            </div>
        </div>
        ''')

        # Input section
        formatted_input = format_input_content(input_content)
        content_sections.append(f'''
        <div class="section input-section">
            <div class="section-header">📥 Input Prompt (Record {i})</div>
            <div class="section-content">
                <div class="code-block">{formatted_input}</div>
            </div>
        </div>
        ''')

        # Output section
        formatted_output = html.escape(output_content)
        content_sections.append(f'''
        <div class="section output-section">
            <div class="section-header">📤 Model Output (Record {i})</div>
            <div class="section-content">
                <div class="code-block">{formatted_output}</div>
            </div>
        </div>
        ''')

        # Reward section
        reward_html = create_reward_section(data)
        content_sections.append(f'''
        <div class="section reward-section">
            <div class="section-header">🏆 Rewards & Scores (Record {i})</div>
            <div class="section-content">
                {reward_html}
            </div>
        </div>
        ''')
        
        # Add separator except for the last record
        if i < total_records:
            content_sections.append('<hr class="record-divider">')
    
    # Calculate average score for the header
    avg_score = sum(data.get('score', 0.0) for data in data_records) / len(data_records) if data_records else 0.0
    
    # Generate HTML
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_content = create_html_template().format(
        step_num=step_num,
        score=f"{avg_score:.4f} (avg of {total_records} records)",
        timestamp=timestamp,
        content=''.join(content_sections)
    )
    
    # Save HTML file
    output_file = output_dir / f"{step_num}.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_file

def create_index_html(output_dir, converted_files):
    """Create an index HTML file listing all converted files"""
    index_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rollout Data Index</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }}
        .file-list {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .file-item {{
            background: #f8f9fa;
            padding: 15px;
            text-align: center;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .file-item:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        .file-item a {{
            color: #1976d2;
            text-decoration: none;
            font-weight: bold;
        }}
        .file-item a:hover {{
            text-decoration: underline;
        }}
        .stats {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Rollout Data Visualization</h1>
            <p>Interactive HTML views of rollout data steps</p>
        </div>
        
        <div class="stats">
            <strong>{total_files}</strong> steps converted | 
            Generated on <strong>{timestamp}</strong>
        </div>
        
        <div class="file-list">
            {file_links}
        </div>
    </div>
</body>
</html>
"""
    
    # Sort files numerically
    file_items = []
    for file in sorted(converted_files, key=lambda x: int(x.stem)):
        file_items.append(f'''
        <div class="file-item">
            <a href="{file.name}">Step {file.stem}</a>
        </div>
        ''')
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    index_html = index_template.format(
        total_files=len(converted_files),
        timestamp=timestamp,
        file_links=''.join(file_items)
    )
    
    index_file = output_dir / "index.html"
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(index_html)
    
    return index_file

def main():
    parser = argparse.ArgumentParser(description='Convert JSONL rollout data to HTML visualization')
    parser.add_argument('--input-dir', default='../sup_rollout_data_dir_1012',
                       help='Input directory containing JSONL files')
    parser.add_argument('--output-dir', default='../sup_rollout_data_html_dir_1012',
                       help='Output directory for HTML files')
    parser.add_argument('--file-pattern', default='*.jsonl', 
                       help='File pattern to match (default: *.jsonl)')
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return 1
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Find all JSONL files
    jsonl_files = list(input_dir.glob(args.file_pattern))
    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}")
        return 1
    
    print(f"Found {len(jsonl_files)} JSONL files")
    
    # Convert files
    converted_files = []
    failed_files = []
    
    for jsonl_file in jsonl_files:
        print(f"Converting {jsonl_file.name}...")
        output_file = convert_jsonl_to_html(jsonl_file, output_dir)
        if output_file:
            converted_files.append(output_file)
        else:
            failed_files.append(jsonl_file)
    
    # Create index file
    if converted_files:
        index_file = create_index_html(output_dir, converted_files)
        print(f"\nCreated index file: {index_file}")
    
    # Summary
    print(f"\n✅ Conversion complete:")
    print(f"  - Successfully converted: {len(converted_files)} files")
    if failed_files:
        print(f"  - Failed to convert: {len(failed_files)} files")
        for f in failed_files:
            print(f"    - {f.name}")
    
    print(f"\nOpen {output_dir / 'index.html'} in your browser to view the results!")
    
    return 0

if __name__ == "__main__":
    exit(main())