# Reward Top Analyzer 更新说明

## 版本更新 - 2025年10月15日

### 新增功能：原始HTML文件集成

#### 功能概述

在分析top奖励样本时，工具现在可以自动复制并集成对应训练步骤的完整HTML文件，方便用户查看top样本所在步骤的完整上下文。

#### 主要改进

1. **新增参数 `--html-dir`**
   - 指定原始HTML文件所在目录
   - 默认值：`../sup_rollout_data_html_dir_1012`
   - 如果目录不存在，工具会给出警告但继续运行

2. **自动复制原始HTML**
   - 收集所有top样本涉及的训练步骤
   - 自动去重，避免重复复制
   - 复制到 `output_dir/original_steps/` 目录
   - 显示复制进度和统计信息

3. **样本详情页面增强**
   - 在页面顶部添加醒目的渐变色按钮
   - 按钮文本：📋 查看完整步骤HTML（包含所有XX步的记录）
   - 点击可跳转到对应步骤的完整HTML页面
   - 仅在原始HTML存在时显示

4. **索引页面增强**
   - 表格新增"原始HTML"列
   - 每个样本都有到完整步骤HTML的直接链接
   - 如果HTML不存在则显示"-"

5. **快捷脚本更新**
   - `analyze_rewards.sh` 增加 `-H/--html-dir` 参数
   - 默认包含HTML目录参数

#### 使用示例

```bash
# 包含原始HTML文件
python3 src/reward_top_analyzer.py \
    --input-dir sup_rollout_data_dir_1012 \
    --output-dir top_rewards_analysis \
    --html-dir sup_rollout_data_html_dir_1012 \
    --start-step 100 \
    --end-step 200

# 或使用快捷脚本（默认已包含HTML目录）
./analyze_rewards.sh -s 100 -e 200
```

#### 输出目录结构

```
top_rewards_analysis/
├── index.html                    # 主索引（包含原始HTML链接列）
├── original_steps/               # 新增：原始HTML文件
│   ├── 100.html                 # 步骤100的完整HTML
│   ├── 101.html
│   └── ...
├── zoom_highest/
│   ├── rank1_stepXX_recYY.html  # 样本详情（包含原始HTML按钮）
│   └── ...
└── ...（其他类别）
```

#### 技术实现

- 使用 `shutil.copy2` 保留文件元数据
- 通过集合去重避免重复复制
- 函数参数使用默认值保持向后兼容
- 在生成HTML时根据文件存在性动态添加链接

#### 优势

1. **完整上下文**：可以查看top样本所在步骤的所有rollout记录
2. **对比分析**：方便对比同一步骤中不同样本的表现
3. **无缝集成**：原始HTML链接直接嵌入分析报告
4. **按需加载**：只复制涉及的步骤，节省空间
5. **向后兼容**：不提供HTML目录时仍可正常工作

#### 文件修改清单

- ✅ `src/reward_top_analyzer.py` - 核心功能实现
- ✅ `analyze_rewards.sh` - 快捷脚本更新
- ✅ `REWARD_ANALYZER_README.md` - 文档更新
- ✅ 测试通过（步骤100-105）

#### 测试结果

- ✅ 参数解析正确
- ✅ HTML文件成功复制（6个文件，约229MB）
- ✅ 样本详情页面显示原始HTML按钮
- ✅ 索引页面显示原始HTML列
- ✅ 链接跳转正常
- ✅ 文件不存在时正确处理

---

更新完成时间：2025年10月15日
