"""
监督学习环境管理器

使用重构后的架构，代码量大幅减少，职责清晰
"""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from uvicorn import run as uvicorn_run
from box import Box

from src.config.environment_config import EnvironmentConfig
from src.core.factory import EnvironmentSystemFactory
from src.api.environment_router import create_environment_router
from src.utils.data_tools import supervise_dataset_construct


# ============ 配置部分 ============
initial_configs = Box({
    'max_num_envs': 8,
    'initial_refresh_env': True,
    'cache_dir': './.auth',
    'env_name': 'classifields',
    'results_dir': '/data/wangzhenchuan/Projects/LIFT/results',
    'max_task_steps': 10,
    'scp_version': 'cmd',
    'type': 'remote',
    'target_server': '192.168.1.5',
    'instruction_path': '/data/wangzhenchuan/Projects/LIFT/visualwebarena/src/prompts/vwa/jsons/lift_d.json',
    'annotate_path': '../data/annotate_with_reasoning_d',
    'annotate_envs': 'classifieds'
})

# ============ 系统初始化 ============
# 1. 加载数据集
dataset = supervise_dataset_construct(initial_configs=initial_configs)

# 2. 创建配置对象
config = EnvironmentConfig(**initial_configs)

# 3. 使用工厂创建整个系统
orchestrator = EnvironmentSystemFactory.create_from_dict(
    config_dict=dict(initial_configs),
    task_configs=dataset
)

# 4. 如果需要，刷新环境
if config.initial_refresh_env:
    orchestrator.refresh_env()


# ============ FastAPI应用 ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理

    启动时初始化环境，关闭时清理资源
    """
    # 启动时：初始化环境
    asyncio.create_task(
        orchestrator.initialize_environments(config.max_num_envs)
    )

    yield  # 控制权交给FastAPI

    # 关闭时的清理逻辑可以在这里添加
    print("应用正在关闭...")


# 创建FastAPI应用
app = FastAPI(lifespan=lifespan, title="监督学习环境管理器")

# 注册路由
router = create_environment_router(orchestrator)
app.include_router(router)


# ============ 主入口 ============
if __name__ == '__main__':
    uvicorn_run(app, host='0.0.0.0', port=7333)
