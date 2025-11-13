#!/bin/bash
# GRPO Rollout 数据可视化转换脚本
# 用于将 sup_rollout_data_dir 中的 JSONL 文件转换为可视化的 HTML 文件

INPUT_DIR="sup_rollout_data_dir"
OUTPUT_DIR="sup_rollout_data_html_dir"
CONVERTER="src/jsonl_to_html_converter.py"

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录 $INPUT_DIR 不存在"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
echo "输出目录: $OUTPUT_DIR"

# 获取文件数量
total_files=$(ls -1 "$INPUT_DIR"/*.jsonl 2>/dev/null | wc -l)
echo "找到 $total_files 个 JSONL 文件"

# 提示用户选择处理模式
echo ""
echo "请选择处理模式:"
echo "1) 处理所有文件 (可能需要较长时间)"
echo "2) 处理前 10 个文件 (快速预览)"
echo "3) 处理指定范围的文件"
echo "4) 处理单个文件"
read -p "请输入选项 (1-4): " mode

case $mode in
    1)
        echo "开始处理所有文件..."
        python3 "$CONVERTER" --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR"
        ;;
    2)
        echo "处理前 10 个文件..."
        count=0
        for file in "$INPUT_DIR"/*.jsonl; do
            [ $count -ge 10 ] && break
            filename=$(basename "$file")
            echo "处理 $filename..."
            python3 -c "
from pathlib import Path
import sys
sys.path.insert(0, 'src')
from jsonl_to_html_converter import convert_jsonl_to_html
convert_jsonl_to_html(Path('$file'), Path('$OUTPUT_DIR'))
"
            ((count++))
        done
        # 创建索引文件
        python3 -c "
from pathlib import Path
import sys
sys.path.insert(0, 'src')
from jsonl_to_html_converter import create_index_html
output_dir = Path('$OUTPUT_DIR')
html_files = list(output_dir.glob('*.html'))
html_files = [f for f in html_files if f.name != 'index.html']
if html_files:
    create_index_html(output_dir, html_files)
"
        ;;
    3)
        read -p "起始文件编号: " start_num
        read -p "结束文件编号: " end_num
        echo "处理文件 $start_num 到 $end_num..."
        for i in $(seq $start_num $end_num); do
            file="$INPUT_DIR/$i.jsonl"
            if [ -f "$file" ]; then
                echo "处理 $i.jsonl..."
                python3 -c "
from pathlib import Path
import sys
sys.path.insert(0, 'src')
from jsonl_to_html_converter import convert_jsonl_to_html
convert_jsonl_to_html(Path('$file'), Path('$OUTPUT_DIR'))
"
            fi
        done
        # 创建索引文件
        python3 -c "
from pathlib import Path
import sys
sys.path.insert(0, 'src')
from jsonl_to_html_converter import create_index_html
output_dir = Path('$OUTPUT_DIR')
html_files = list(output_dir.glob('*.html'))
html_files = [f for f in html_files if f.name != 'index.html']
if html_files:
    create_index_html(output_dir, html_files)
"
        ;;
    4)
        read -p "请输入文件编号: " file_num
        file="$INPUT_DIR/$file_num.jsonl"
        if [ -f "$file" ]; then
            echo "处理 $file_num.jsonl..."
            python3 -c "
from pathlib import Path
import sys
sys.path.insert(0, 'src')
from jsonl_to_html_converter import convert_jsonl_to_html
convert_jsonl_to_html(Path('$file'), Path('$OUTPUT_DIR'))
"
        else
            echo "错误: 文件 $file 不存在"
            exit 1
        fi
        ;;
    *)
        echo "无效的选项"
        exit 1
        ;;
esac

echo ""
echo "✅ 转换完成!"
echo "在浏览器中打开 $OUTPUT_DIR/index.html 查看结果"
