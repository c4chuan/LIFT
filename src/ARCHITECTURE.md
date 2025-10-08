# 重构后的系统架构

## 📐 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EnvironmentRouter (API层)                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  GET  /get_messages         POST /feed_responses        │  │
│  │  GET  /get_length           POST /get_valid_action_rewards│  │
│  │  GET  /refresh_env          GET  /status                │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              EnvironmentOrchestrator (业务协调层)                │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  • initialize_environments()                            │   │
│  │  • parallel_produce()                                   │   │
│  │  • feed_responses()                                     │   │
│  │  • get_messages()                                       │   │
│  │  • get_valid_action_rewards()                          │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                  │
│  依赖组件：                                                      │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │TaskPool  │  │MessageBuilder│  │RewardCalc.   │            │
│  └──────────┘  └──────────────┘  └──────────────┘            │
│  ┌──────────────────┐  ┌──────────────────┐                  │
│  │ActionStrategy    │  │EnvironmentConfig │                  │
│  └──────────────────┘  └──────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │   TaskPool   │ │ MessageBuilder│ │RewardCalc   │
    └──────────────┘ └──────────────┘ └──────────────┘
            │
            ▼
    ┌──────────────────────────────────┐
    │       VWATask (数据模型)          │
    │  ┌────────────────────────────┐  │
    │  │ • task_id                  │  │
    │  │ • env (BrowserEnv)         │  │
    │  │ • state_trajectory         │  │
    │  │ • action_history           │  │
    │  │ • ref_trajectory (可选)     │  │
    │  │ • state (TaskState)        │  │
    │  └────────────────────────────┘  │
    └──────────────────────────────────┘
```

## 🔄 数据流

### 1. 初始化流程
```
User Start
    │
    ▼
EnvironmentSystemFactory.create_from_dict()
    │
    ├─► EnvironmentConfig (验证配置)
    │
    ├─► TaskPool (创建任务池)
    │
    ├─► MessageBuilder (创建消息构建器)
    │
    ├─► RewardCalculator (创建奖励计算器)
    │
    ├─► ActionStrategy (根据模式选择策略)
    │
    └─► EnvironmentOrchestrator (组装所有组件)
         │
         ▼
    orchestrator.initialize_environments(N)
         │
         ├─► 创建N个VWATask实例
         │
         ├─► 并行reset所有环境
         │
         └─► 生成初始消息 → 加入message_queue
```

### 2. 消息生产流程
```
parallel_produce(tasks, actions)
    │
    ├─► 判断任务是否需要reset
    │    │
    │    ├─► 需要reset: parallel_prepare() + env.areset()
    │    └─► 不需要reset: ActionStrategy.decide_action() + env.astep()
    │
    ├─► asyncio.gather() 并行执行
    │
    ├─► 处理结果
    │    │
    │    ├─► 构建state_info
    │    ├─► MessageBuilder.construct_message()
    │    └─► 更新task状态
    │
    ├─► 发送图片到服务器 (可选)
    │
    └─► TaskPool.enqueue_message() → message_queue
```

### 3. 消息消费流程
```
Client: GET /get_messages?num=N
    │
    ▼
orchestrator.get_messages(N)
    │
    ├─► 检查message_queue长度
    │
    ├─► TaskPool.dequeue_messages(N)
    │    │
    │    ├─► 取出N条消息
    │    └─► 标记对应任务为PROCESSING状态
    │
    └─► 返回消息给客户端

Client: 处理消息 → 获得响应

Client: POST /feed_responses
    │
    ▼
orchestrator.feed_responses(responses)
    │
    ├─► MessageBuilder.extract_action() → 解析动作
    │
    ├─► 更新任务信息
    │    ├─► task.steps += 1
    │    ├─► MessageBuilder.extract_summary()
    │    └─► 更新状态: PROCESSING → BUSY
    │
    └─► asyncio.create_task(parallel_produce())
         │
         └─► 异步生产新消息
```

### 4. 奖励计算流程
```
Client: POST /get_valid_action_rewards
    │
    ▼
orchestrator.get_valid_action_rewards(responses)
    │
    ├─► 获取PROCESSING状态的任务
    │
    ├─► 按batch_size分组
    │
    └─► RewardCalculator.calculate_batch_rewards()
         │
         ├─► 检查格式 (action_format_reward)
         │
         ├─► 提取动作信息 (get_action_id_answer)
         │
         └─► 计算奖励
              │
              ├─► 验证element_id有效性
              ├─► 评估点击对齐度
              └─► 返回奖励值
```

## 🏗️ 组件详解

### TaskPool (任务池管理器)

**职责**: 管理所有任务实例的生命周期和状态

**核心数据结构**:
```python
_tasks: Dict[int, VWATask]                    # {task_id: task}
_state_index: Dict[TaskState, Set[int]]       # {state: {task_ids}}
_message_queue: List[MessageQueueItem]        # 消息队列
```

**关键方法**:
- `create_task()`: 创建新任务实例
- `update_task_state()`: 线程安全的状态转换
- `get_tasks_by_state()`: O(1)状态查询
- `enqueue_message()` / `dequeue_messages()`: 队列操作

**优势**:
- ✅ O(1)查找复杂度（字典替代列表）
- ✅ 自动维护状态索引
- ✅ 线程安全（asyncio.Lock）

### MessageBuilder (消息构建器)

**职责**: 统一的消息构建接口

**接口定义**:
```python
class IMessageBuilder(ABC):
    def construct_message(task, obs, info) -> List[Dict]
    def extract_action(response) -> str
    def extract_summary(response) -> str
```

**实现**:
- `EnvLIFTMessageBuilder`: 封装EnvLIFTConstructor

**优势**:
- ✅ 接口抽象，易于扩展
- ✅ 统一消息格式
- ✅ 可插拔设计

### ActionStrategy (动作策略)

**职责**: 决定使用哪个动作（当前动作 vs 参考动作）

**策略类型**:
```python
StandardActionStrategy       # 直接使用当前动作
SupervisedActionStrategy     # 可能使用参考动作
AdaptiveActionStrategy       # 自适应调整
```

**关键方法**:
```python
def decide_action(task, current_action) -> action
```

**优势**:
- ✅ 策略模式实现
- ✅ 监督学习/标准模式统一
- ✅ 易于扩展新策略

### RewardCalculator (奖励计算器)

**职责**: 验证动作有效性并计算奖励

**计算流程**:
```
response → 格式验证 → 提取动作信息 → 计算奖励
    │
    ├─► element_id验证 → 范围检查
    ├─► 点击对齐度 → summary匹配
    └─► answer验证 → evaluator (TODO)
```

**优势**:
- ✅ 独立模块，易于测试
- ✅ 清晰的奖励逻辑
- ✅ 支持批量计算

### EnvironmentOrchestrator (核心协调器)

**职责**: 协调所有组件，实现核心业务流程

**依赖注入**:
```python
def __init__(
    config: EnvironmentConfig,
    task_pool: TaskPool,
    message_builder: IMessageBuilder,
    reward_calculator: RewardCalculator,
    action_strategy: IActionStrategy
)
```

**核心流程**:
- `parallel_produce()`: 并行环境交互
- `feed_responses()`: 处理模型响应
- `get_messages()`: 获取消息队列

**优势**:
- ✅ 单一职责（仅协调）
- ✅ 依赖注入（易测试）
- ✅ 异步并发控制

## 🎯 设计原则应用

### SOLID原则

| 原则 | 应用 |
|------|------|
| **S**RP | 每个类只负责一个功能模块 |
| **O**CP | 通过策略模式和接口扩展 |
| **L**SP | 所有策略都可替换使用 |
| **I**SP | 接口职责单一（IMessageBuilder） |
| **D**IP | 依赖抽象接口而非具体实现 |

### 其他设计原则

- **DRY**: 消除95%代码重复
- **KISS**: 每个组件保持简单
- **YAGNI**: 只实现必要功能
- **组合优于继承**: 通过组合实现复用

## 🔧 可扩展性

### 添加新功能示例

#### 1. 添加新的奖励策略
```python
class CustomRewardCalculator(RewardCalculator):
    def calculate_single_reward(self, response, task):
        # 自定义奖励逻辑
        return reward
```

#### 2. 添加新的消息格式
```python
class CustomMessageBuilder(IMessageBuilder):
    def construct_message(self, task, obs, info):
        # 自定义消息格式
        return custom_message
```

#### 3. 添加新的API端点
```python
@router.get('/custom_endpoint')
async def custom_endpoint():
    # 自定义逻辑
    return result
```

## 📊 性能优化

### 已实现的优化

1. **O(1)任务查找**: 字典替代列表
2. **状态索引**: 快速按状态查询
3. **异步并发**: asyncio.gather()并行执行
4. **事件驱动**: 替代忙等待

### 未来优化方向

1. **对象池**: 环境实例复用
2. **批处理**: 更大的批量操作
3. **缓存**: 消息缓存机制
4. **连接池**: 数据库/Redis连接池

---

**架构版本**: v2.0
**最后更新**: 2025-10-08
