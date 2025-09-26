#!/usr/bin/env python3
"""
测试EnvironmentController的SOM截图保存和访问功能
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# 添加路径以便导入模块
sys.path.append(str(Path(__file__).parent))

from src.annotation.environment_controller import EnvironmentController


async def test_screenshot_functionality():
    """测试截图功能"""
    print("🚀 开始测试SOM截图保存和访问功能...")

    # 创建临时测试配置
    test_config = {
        "task_id": "test_task_123",
        "intent": "测试SOM截图保存功能",
        "start_url": "https://www.example.com",
        "sites": ["example"],
        "require_login": False,
    }

    # 创建临时配置文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f, ensure_ascii=False, indent=2)
        temp_config_path = f.name

    try:
        # 创建环境控制器
        env_controller = EnvironmentController(
            render=False,  # 无头模式测试
            result_dir="data/test_results"
        )

        print("✅ EnvironmentController 创建成功")

        # 测试初始化环境
        try:
            print("📦 初始化测试环境...")
            state_info, task_info = await env_controller.initialize_environment(temp_config_path)
            print(f"✅ 环境初始化成功，Task ID: {task_info['task_id']}")

            # 测试截图信息获取
            screenshot_info = env_controller.get_screenshot_info()
            print(f"📊 截图信息: {screenshot_info}")

            # 测试截图目录
            screenshot_dir = env_controller.get_task_screenshot_directory()
            print(f"📂 截图目录: {screenshot_dir}")

            # 验证截图目录是否创建
            if screenshot_dir and os.path.exists(screenshot_dir):
                print("✅ 截图目录创建成功")

                # 列出目录内容
                files = list(Path(screenshot_dir).glob("*"))
                print(f"📁 目录内容: {[f.name for f in files]}")

                # 测试获取初始截图
                initial_screenshot = env_controller.get_screenshot_path(0)
                if initial_screenshot and os.path.exists(initial_screenshot):
                    print(f"✅ 初始截图存在: {initial_screenshot}")

                    # 检查文件大小
                    file_size = os.path.getsize(initial_screenshot)
                    print(f"📏 初始截图文件大小: {file_size} bytes")
                else:
                    print("❌ 初始截图未找到")
            else:
                print("❌ 截图目录创建失败")

            # 测试访问接口
            print("\n🔍 测试访问接口:")
            print(f"- 获取所有截图: {len(env_controller.get_all_screenshots())} 个")
            print(f"- 截图总数: {env_controller.get_screenshot_count()}")
            print(f"- 最新截图: {env_controller.get_latest_screenshot_path()}")
            print(f"- 当前截图: {env_controller.get_current_screenshot_path()}")

            # 测试保存轨迹数据
            print("\n💾 测试轨迹数据保存...")
            trajectory_file = env_controller.save_trajectory_data()
            if trajectory_file and os.path.exists(trajectory_file):
                print(f"✅ 轨迹数据保存成功: {trajectory_file}")

                # 读取并验证轨迹数据
                with open(trajectory_file, 'r', encoding='utf-8') as f:
                    trajectory_data = json.load(f)
                    print(f"📊 轨迹数据包含 {len(trajectory_data['trajectory'])} 个项目")
                    print(f"🆔 Task ID: {trajectory_data['task_id']}")
                    print(f"📸 截图路径: {len(trajectory_data['screenshot_paths'])} 个")
            else:
                print("❌ 轨迹数据保存失败")

            # 关闭环境
            await env_controller.close_environment()
            print("✅ 环境关闭成功")

        except Exception as e:
            print(f"❌ 测试过程中发生错误: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # 清理环境
            try:
                await env_controller.close_environment()
            except:
                pass

    finally:
        # 删除临时配置文件
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)
            print("🧹 清理临时文件完成")

    print("\n🎉 SOM截图功能测试完成！")


async def test_mock_screenshot_functionality():
    """使用模拟数据测试截图功能（不需要真实浏览器环境）"""
    print("🧪 开始模拟数据测试...")

    try:
        # 创建环境控制器
        env_controller = EnvironmentController(
            render=False,
            result_dir="data/test_results"
        )

        # 手动设置一些测试属性
        env_controller.current_task_id = "mock_test_456"
        env_controller.task_screenshot_dir = Path("data/test_results") / "task_mock_test_456"
        env_controller.task_screenshot_dir.mkdir(parents=True, exist_ok=True)
        env_controller.screenshot_paths = []

        # 创建模拟状态信息
        import numpy as np
        mock_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        mock_state_info = {
            "observation": {
                "image": mock_image,
                "text": "这是一个模拟的观察结果"
            },
            "info": {"test": "mock_info"},
            "url": "https://test.com"
        }

        # 测试保存截图
        print("💾 测试保存模拟截图...")
        screenshot_path = env_controller._save_som_screenshot(mock_state_info, is_initial=True)

        if screenshot_path and os.path.exists(screenshot_path):
            print(f"✅ 模拟截图保存成功: {screenshot_path}")

            # 验证文件大小
            file_size = os.path.getsize(screenshot_path)
            print(f"📏 模拟截图文件大小: {file_size} bytes")

            # 测试访问接口
            env_controller.screenshot_paths.append(screenshot_path)
            print(f"🔍 截图总数: {env_controller.get_screenshot_count()}")
            print(f"📸 最新截图: {env_controller.get_latest_screenshot_path()}")

        else:
            print("❌ 模拟截图保存失败")

        print("✅ 模拟数据测试完成")

    except Exception as e:
        print(f"❌ 模拟测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("🔧 SOM截图功能测试工具")
    print("=" * 50)

    # 运行模拟测试（不需要浏览器环境）
    asyncio.run(test_mock_screenshot_functionality())

    print("\n" + "=" * 50)
    print("💡 如需完整测试，请确保浏览器环境配置正确，然后取消注释下面的代码")
    print("💡 完整测试需要有效的浏览器配置和网络连接")

    # 完整测试（需要浏览器环境，默认注释掉）
    # asyncio.run(test_screenshot_functionality())


if __name__ == "__main__":
    main()