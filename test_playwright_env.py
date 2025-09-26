#!/usr/bin/env python3
"""
Playwright环境测试脚本

用于诊断和验证Playwright安装状态，排查CodeGen启动问题。
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def test_playwright_installation():
    """测试Playwright是否正确安装"""
    print("=== 测试Playwright安装状态 ===")

    try:
        # 检查playwright命令是否可用
        result = subprocess.run(
            ["playwright", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            print(f"OK Playwright已安装: {result.stdout.strip()}")
            return True
        else:
            print(f"ERROR Playwright命令执行失败: {result.stderr}")
            return False

    except FileNotFoundError:
        print("ERROR Playwright命令未找到，请确保已安装playwright")
        return False
    except subprocess.TimeoutExpired:
        print("ERROR Playwright命令执行超时")
        return False
    except Exception as e:
        print(f"ERROR 检查Playwright安装时出错: {e}")
        return False


def test_playwright_browsers():
    """测试Playwright浏览器是否已安装"""
    print("\n=== 测试Playwright浏览器安装状态 ===")

    try:
        result = subprocess.run(
            ["playwright", "install", "--dry-run"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            print("OK 浏览器安装状态检查完成")
            if "already installed" in result.stdout.lower():
                print("OK 浏览器已安装")
                return True
            else:
                print("WARN 可能需要安装浏览器，运行: playwright install")
                print(f"详细信息: {result.stdout}")
                return False
        else:
            print(f"ERROR 浏览器状态检查失败: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("ERROR 浏览器检查超时")
        return False
    except Exception as e:
        print(f"ERROR 检查浏览器时出错: {e}")
        return False


def test_codegen_basic():
    """测试基本的playwright codegen命令"""
    print("\n=== 测试基本CodeGen命令 ===")

    try:
        # 创建临时输出文件
        temp_output = "temp_test_codegen.py"

        # 构建最简单的codegen命令
        command = [
            "playwright", "codegen",
            "--target", "python",
            "-o", temp_output,
            "--timeout", "5000",  # 5秒超时
            "https://www.baidu.com"
        ]

        print(f"🔧 测试命令: {' '.join(command)}")

        # 启动进程但不等待用户交互
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # 等待3秒看是否能正常启动
        import time
        time.sleep(3)

        if process.poll() is None:
            print("OK CodeGen进程启动成功")
            # 终止进程
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

            # 清理临时文件
            if os.path.exists(temp_output):
                os.remove(temp_output)

            return True
        else:
            # 进程已经退出
            stdout, stderr = process.communicate()
            print(f"ERROR CodeGen进程立即退出")
            print(f"   退出代码: {process.returncode}")
            if stdout:
                print(f"   标准输出: {stdout}")
            if stderr:
                print(f"   错误输出: {stderr}")

            return False

    except Exception as e:
        print(f"ERROR 测试CodeGen时出错: {e}")
        return False


def test_storage_state_files():
    """检查storage_state文件"""
    print("\n=== 检查Storage State文件 ===")

    storage_files = [
        "src/.auth/classifieds_state.json",
        "src/.auth/reddit_state.json",
        "src/.auth/shopping_state.json"
    ]

    all_found = True
    for file_path in storage_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    print(f"OK 找到storage文件: {file_path} (包含 {len(data.get('cookies', []))} 个cookies)")
            except Exception as e:
                print(f"WARN storage文件损坏: {file_path} - {e}")
                all_found = False
        else:
            print(f"ERROR 缺少storage文件: {file_path}")
            all_found = False

    return all_found


def test_output_directories():
    """检查输出目录权限"""
    print("\n=== 检查输出目录 ===")

    dirs_to_check = [
        "data/annotation_progress/temp",
        "data/human_trajectories",
        "logs"
    ]

    all_good = True
    for dir_path in dirs_to_check:
        try:
            os.makedirs(dir_path, exist_ok=True)
            # 测试写权限
            test_file = os.path.join(dir_path, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"OK 目录权限正常: {dir_path}")
        except Exception as e:
            print(f"ERROR 目录权限问题: {dir_path} - {e}")
            all_good = False

    return all_good


def test_task_files():
    """检查任务文件"""
    print("\n=== 检查任务文件 ===")

    task_files = [
        "data/annotate/classifieds_tasks.json",
        "data/annotate/reddit_tasks.json",
        "data/annotate/shopping_tasks.json"
    ]

    total_tasks = 0
    for file_path in task_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    task_count = len(data)
                    total_tasks += task_count
                    print(f"OK 任务文件: {file_path} ({task_count} 个任务)")
            except Exception as e:
                print(f"ERROR 任务文件损坏: {file_path} - {e}")
        else:
            print(f"ERROR 缺少任务文件: {file_path}")

    print(f"INFO 总任务数: {total_tasks}")
    return total_tasks > 0


def run_comprehensive_test():
    """运行全面的环境测试"""
    print("开始Playwright环境诊断...\n")

    tests = [
        ("Playwright安装", test_playwright_installation),
        ("浏览器安装", test_playwright_browsers),
        ("基本CodeGen", test_codegen_basic),
        ("Storage文件", test_storage_state_files),
        ("输出目录", test_output_directories),
        ("任务文件", test_task_files)
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"ERROR 测试 {name} 时出错: {e}")
            results[name] = False

    print("\n" + "="*50)
    print("诊断结果汇总:")
    print("="*50)

    for name, result in results.items():
        status = "正常" if result else "异常"
        print(f"{name}: {status}")

    all_passed = all(results.values())
    print(f"\n总体状态: {'环境正常' if all_passed else '发现问题'}")

    if not all_passed:
        print("\n建议修复步骤:")
        if not results.get("Playwright安装"):
            print("- 运行: pip install playwright")
        if not results.get("浏览器安装"):
            print("- 运行: playwright install")
        if not results.get("Storage文件"):
            print("- 检查 src/.auth/ 目录下的登录状态文件")
        if not results.get("输出目录"):
            print("- 检查目录权限，确保可以写入文件")
        if not results.get("任务文件"):
            print("- 检查 data/annotate/ 目录下的任务JSON文件")

    return all_passed


if __name__ == "__main__":
    run_comprehensive_test()