#!/usr/bin/env python3
"""
自动登录测试脚本

用于测试自动登录功能的独立脚本
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.auto_login import AutoLogin


def test_login_status_check():
    """测试登录状态检查"""
    print("=== 测试登录状态检查 ===")

    auto_login = AutoLogin()

    sites_to_test = ["classifieds", "reddit", "shopping"]

    for site in sites_to_test:
        print(f"\n测试网站: {site}")
        try:
            is_expired = auto_login.is_login_expired(site)
            print(f"登录状态: {'已过期' if is_expired else '有效'}")
        except Exception as e:
            print(f"检查失败: {e}")


def test_auto_login():
    """测试自动登录功能"""
    print("\n=== 测试自动登录功能 ===")

    auto_login = AutoLogin()

    # 只测试 classifieds，因为它的服务器通常在运行
    site_to_test = "classifieds"

    print(f"测试自动登录: {site_to_test}")
    try:
        success = auto_login.perform_login(site_to_test, headless=True)
        print(f"自动登录结果: {'成功' if success else '失败'}")

        if success:
            # 再次检查登录状态
            print("验证登录状态...")
            is_expired = auto_login.is_login_expired(site_to_test)
            print(f"登录验证: {'失败' if is_expired else '成功'}")

    except Exception as e:
        print(f"自动登录失败: {e}")


def test_batch_login():
    """测试批量登录检查"""
    print("\n=== 测试批量登录检查 ===")

    auto_login = AutoLogin()
    sites = ["classifieds", "reddit", "shopping"]

    try:
        print("执行批量登录检查...")
        all_valid = auto_login.ensure_login_valid(sites)
        print(f"批量检查结果: {'全部有效' if all_valid else '部分失效'}")
    except Exception as e:
        print(f"批量检查失败: {e}")


def main():
    """主测试函数"""
    print("开始自动登录功能测试...\n")

    try:
        # 测试登录状态检查
        test_login_status_check()

        # 测试自动登录
        test_auto_login()

        # 测试批量登录检查
        test_batch_login()

        print("\n=== 测试完成 ===")

    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()