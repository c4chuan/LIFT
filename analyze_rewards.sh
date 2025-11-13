#!/bin/bash
# Reward Top Analyzer - 快捷启动脚本

# 默认参数
INPUT_DIR="sup_rollout_data_dir_1012"
OUTPUT_DIR="top_rewards_analysis"
HTML_DIR="sup_rollout_data_html_dir_1012"
START_STEP=""
END_STEP=""

# 显示帮助信息
show_help() {
    cat << EOF
Reward Top Analyzer - 快捷启动脚本

用法: $0 [选项]

选项:
    -i, --input-dir DIR     输入目录（默认: sup_rollout_data_dir_1012）
    -o, --output-dir DIR    输出目录（默认: top_rewards_analysis）
    -H, --html-dir DIR      原始HTML目录（默认: sup_rollout_data_html_dir_1012）
    -s, --start-step NUM    起始步骤（可选）
    -e, --end-step NUM      结束步骤（可选）
    -h, --help              显示此帮助信息

示例:
    # 分析所有数据
    $0

    # 分析步骤100-200
    $0 -s 100 -e 200

    # 指定输入输出目录
    $0 -i my_data -o my_results

    # 完整示例
    $0 -i sup_rollout_data_dir_1012 -o results -s 100 -e 200
EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -H|--html-dir)
            HTML_DIR="$2"
            shift 2
            ;;
        -s|--start-step)
            START_STEP="$2"
            shift 2
            ;;
        -e|--end-step)
            END_STEP="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 构建命令
CMD="python3 src/reward_top_analyzer.py --input-dir $INPUT_DIR --output-dir $OUTPUT_DIR"

if [ -n "$HTML_DIR" ]; then
    CMD="$CMD --html-dir $HTML_DIR"
fi

if [ -n "$START_STEP" ]; then
    CMD="$CMD --start-step $START_STEP"
fi

if [ -n "$END_STEP" ]; then
    CMD="$CMD --end-step $END_STEP"
fi

# 显示即将执行的命令
echo "执行命令: $CMD"
echo ""

# 执行命令
eval $CMD

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 分析完成！"
    echo "📂 结果目录: $OUTPUT_DIR"
    echo "🌐 用浏览器打开: $OUTPUT_DIR/index.html"
    echo "=========================================="
else
    echo ""
    echo "❌ 分析失败，请检查错误信息"
    exit 1
fi
