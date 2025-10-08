# 快速开始指南

## 🚀 启动服务

### 方式1: 标准模式（无监督学习）

```bash
# 设置环境变量
export DATASET=classifieds

# 启动服务
python src/environment_manager.py
```

### 方式2: 监督学习模式

```bash
# 设置环境变量
export DATASET=classifieds

# 启动服务
python src/supervise_env_manager.py
```

服务将在 `http://0.0.0.0:7333` 启动

## 📡 API使用示例

### 1. 获取任务总数
```bash
curl http://localhost:7333/get_length
# 返回: {"length": 100}
```

### 2. 获取消息
```bash
curl "http://localhost:7333/get_messages?num=2"
# 返回: {"messages": [...]}
```

### 3. 提交响应
```bash
curl -X POST http://localhost:7333/feed_responses \
  -H "Content-Type: application/json" \
  -d '["response1", "response2"]'
# 返回: {"status": "accepted"}
```

### 4. 获取奖励
```bash
curl -X POST http://localhost:7333/get_valid_action_rewards \
  -H "Content-Type: application/json" \
  -d '{"responses": ["response1", "response2"]}'
# 返回: {"rewards": [0.0, 0.5, -1.0]}
```

### 5. 查看系统状态
```bash
curl http://localhost:7333/status
# 返回系统详细状态
```

### 6. 刷新环境
```bash
curl http://localhost:7333/refresh_env
# 返回: {"status": "refreshed"}
```

## 🔧 配置说明

### 修改配置

编辑 `environment_manager.py` 或 `supervise_env_manager.py` 中的配置：

```python
initial_configs = Box({
    'max_num_envs': 8,              # 并发环境数量
    'max_task_steps': 4,             # 每个任务最大步数
    'env_name': 'classifieds',       # 环境名称
    'cache_dir': './.auth',          # 缓存目录
    'results_dir': './results',      # 结果目录
    'scp_version': 'cmd',            # SCP版本: 'client' 或 'cmd'
    'type': 'remote',                # 'local' 或 'remote'
    'target_server': '192.168.1.5',  # 远程服务器地址
    # 监督学习专用配置
    'annotate_path': '../data/annotate/trajectories',
    'annotate_envs': 'classifieds'
})
```

### 配置验证

所有配置都通过Pydantic自动验证：

```python
from src.config.environment_config import EnvironmentConfig

# 自动验证类型和范围
config = EnvironmentConfig(
    max_num_envs=8,     # 必须在 1-32 之间
    max_task_steps=4,   # 必须在 1-20 之间
    # ... 其他配置
)
```

## 🎯 典型工作流

### 完整的训练循环

```python
import requests
import time

BASE_URL = "http://localhost:7333"

# 1. 获取任务总数
response = requests.get(f"{BASE_URL}/get_length")
total_tasks = response.json()['length']
print(f"总任务数: {total_tasks}")

# 2. 循环处理任务
batch_size = 2
for iteration in range(100):
    # 2.1 获取消息
    response = requests.get(
        f"{BASE_URL}/get_messages",
        params={"num": batch_size}
    )
    messages = response.json()['messages']

    if messages == "Finished":
        print("所有任务已完成")
        break

    if not messages:
        print("等待消息生产...")
        time.sleep(1)
        continue

    # 2.2 使用你的模型处理消息
    model_responses = your_model.generate(messages)

    # 2.3 提交响应
    requests.post(
        f"{BASE_URL}/feed_responses",
        json=model_responses
    )

    # 2.4 (可选) 计算奖励
    response = requests.post(
        f"{BASE_URL}/get_valid_action_rewards",
        json={"responses": model_responses}
    )
    rewards = response.json()['rewards']
    print(f"Iteration {iteration}, Rewards: {rewards}")

    time.sleep(2)  # 等待环境处理
```

## 🧪 自定义扩展

### 添加自定义动作策略

```python
from src.core.action_strategy import IActionStrategy
from src.core.factory import EnvironmentSystemFactory

class MyCustomStrategy(IActionStrategy):
    def decide_action(self, task, current_action):
        # 你的自定义逻辑
        if some_condition(task):
            return current_action
        else:
            return task.get_ref_action_at_index(task.get_current_action_index())

# 使用自定义策略
from src.config import EnvironmentConfig
from src.core import TaskPool, EnvironmentOrchestrator
# ... 其他导入

orchestrator = EnvironmentOrchestrator(
    config=config,
    task_pool=task_pool,
    message_builder=message_builder,
    reward_calculator=reward_calculator,
    action_strategy=MyCustomStrategy()  # 使用自定义策略
)
```

### 添加自定义奖励计算

```python
from src.core.reward_calculator import RewardCalculator

class MyRewardCalculator(RewardCalculator):
    def calculate_single_reward(self, response, task):
        # 你的自定义奖励逻辑
        base_reward = super().calculate_single_reward(response, task)

        # 添加额外的奖励项
        bonus = self._calculate_bonus(response, task)

        return base_reward + bonus

    def _calculate_bonus(self, response, task):
        # 自定义bonus逻辑
        return 0.1 if len(response) > 100 else 0.0
```

### 添加自定义API端点

```python
from fastapi import APIRouter
from src.api.environment_router import create_environment_router

# 获取默认路由
router = create_environment_router(orchestrator)

# 添加自定义端点
@router.get('/custom_metric')
async def get_custom_metric():
    # 自定义逻辑
    metric = calculate_some_metric(orchestrator)
    return {"metric": metric}

# 注册到app
app.include_router(router)
```

## 📊 监控和调试

### 查看实时状态

```python
import requests
import json

response = requests.get("http://localhost:7333/status")
status = response.json()

print(json.dumps(status, indent=2))
# 输出:
# {
#   "task_pool": {
#     "total_configs": 100,
#     "active_tasks": 8,
#     "task_pointer": 15,
#     "state_counts": {
#       "idle": 6,
#       "busy": 0,
#       "processing": 2
#     },
#     "queue_length": 4
#   },
#   "config": {
#     "max_num_envs": 8,
#     "max_task_steps": 4,
#     "env_name": "classifieds"
#   }
# }
```

### 日志输出

系统会自动输出关键信息：

```
正在初始化8个环境...
环境初始化完成，创建了8个环境
Task:0-Action:NONEstep完毕-加入消息队列
状态统计: {'idle': 8}, 消息队列: 8
取出2条消息
状态统计: {'idle': 6, 'processing': 2}, 消息队列: 6
```

## 🐛 常见问题

### Q: KeyError: 'DATASET'
**A:** 需要设置环境变量：
```bash
export DATASET=classifieds
```

### Q: 消息队列为空
**A:** 等待环境生产消息，或检查busy_tasks状态

### Q: 如何切换监督学习模式？
**A:**
1. 使用 `supervise_env_manager.py` 而非 `environment_manager.py`
2. 确保配置中设置了 `annotate_path`

### Q: 性能优化建议？
**A:**
1. 增加 `max_num_envs` 提高并发
2. 调整 `max_workers` 参数（在EnvironmentConfig中）
3. 使用 `scp_version='cmd'` 而非 'client'

## 📚 更多文档

- [架构文档](ARCHITECTURE.md) - 详细的系统架构说明
- [重构总结](REFACTORING_SUMMARY.md) - 重构过程和收益

---

**最后更新**: 2025-10-08
