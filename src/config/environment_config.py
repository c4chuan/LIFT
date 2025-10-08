"""
环境配置管理模块

使用Pydantic进行类型验证和配置管理
"""
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


class EnvironmentConfig(BaseModel):
    """环境管理器配置类"""

    # 环境相关配置
    max_num_envs: int = Field(default=2, ge=1, le=32, description="最大并发环境数量")
    env_name: str = Field(default="classifieds", description="环境名称")
    initial_refresh_env: bool = Field(default=False, description="启动时是否刷新环境")

    # 路径配置
    cache_dir: str = Field(default="./.auth", description="缓存目录")
    results_dir: str = Field(default="./results", description="结果保存目录")
    instruction_path: str = Field(
        default="./visualwebarena/src/prompts/vwa/jsons/lift.json",
        description="指令模板路径"
    )

    # 任务配置
    max_task_steps: int = Field(default=4, ge=1, le=20, description="每个任务的最大步数")
    task_pointer_limit: int = Field(default=200, ge=1, description="任务指针循环限制")

    # SCP配置
    scp_version: Literal["client", "cmd"] = Field(default="cmd", description="SCP版本")
    type: Literal["local", "remote"] = Field(default="remote", description="运行类型")
    target_server: Optional[str] = Field(default=None, description="目标服务器地址")

    # 监督学习配置
    annotate_path: Optional[str] = Field(default=None, description="标注轨迹文件夹路径")
    annotate_envs: Optional[str] = Field(default=None, description="标注的环境，多个用空格分隔")

    # 环境实例配置
    headless: bool = Field(default=True, description="是否使用无头模式")
    slow_mo: int = Field(default=0, ge=0, description="慢动作延迟（毫秒）")
    action_set_tag: str = Field(default="som", description="动作集标签")
    observation_type: str = Field(
        default="image_som_without_caption",
        description="观察类型"
    )
    current_viewport_only: bool = Field(default=True, description="仅使用当前视口")
    viewport_width: int = Field(default=1280, ge=800, le=1920, description="视口宽度")
    viewport_height: int = Field(default=2048, ge=600, le=4096, description="视口高度")
    save_trace_enabled: bool = Field(default=False, description="是否保存追踪")
    sleep_after_execution: float = Field(default=2.5, ge=0, description="执行后等待时间（秒）")

    # 并发控制配置
    max_workers: int = Field(default=8, ge=1, le=32, description="并行准备任务的最大工作线程数")
    async_timeout: float = Field(default=30.0, ge=1, description="异步操作超时时间（秒）")
    retry_max_attempts: int = Field(default=3, ge=1, description="最大重试次数")
    retry_delay: float = Field(default=1.0, ge=0, description="重试延迟（秒）")

    @field_validator("target_server")
    @classmethod
    def validate_target_server(cls, v, info):
        """验证远程模式必须提供target_server"""
        if info.data.get("type") == "remote" and not v:
            raise ValueError("远程模式必须提供 target_server")
        return v

    @field_validator("annotate_path")
    @classmethod
    def validate_annotate_path(cls, v, info):
        """验证标注路径"""
        if v and not v.strip():
            return None
        return v

    class Config:
        """Pydantic配置"""
        frozen = False  # 允许修改
        validate_assignment = True  # 赋值时验证
        extra = "forbid"  # 禁止额外字段

    def get_viewport_size(self) -> dict:
        """获取视口大小配置"""
        return {
            "width": self.viewport_width,
            "height": self.viewport_height
        }

    def is_supervised_mode(self) -> bool:
        """判断是否为监督学习模式"""
        return self.annotate_path is not None and self.annotate_path.strip() != ""
