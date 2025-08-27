#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

def run_shell_script(script_name: str) -> Dict[str, Any]:
    """
    执行一个 shell 脚本，返回结果字典，不会直接退出进程。

    :param script_name: 要执行的脚本文件名（例如 "myscript.sh"）。
    :return: 包含脚本名、状态、stdout、stderr、returncode 的字典
    """
    script_path = os.path.join(os.getcwd(), script_name)

    if not os.path.isfile(script_path):
        return {
            "script": script_name,
            "status": "NOT_FOUND",
            "stdout": "",
            "stderr": f"脚本文件 '{script_name}' 不存在。当前目录：{os.getcwd()}",
            "returncode": None
        }

    # 确保可执行权限
    if not os.access(script_path, os.X_OK):
        os.chmod(script_path, os.stat(script_path).st_mode | 0o111)

    try:
        script_output_path = os.path.join(os.getcwd(), f"{script_name}.log")
        with open(script_output_path, 'w', encoding='utf-8') as log_file:
            pass
        log_file = open(script_output_path, 'a', encoding='utf-8')
        # 使用 Popen 执行脚本
        proc = subprocess.Popen(
            ["sh", script_path],
            # shell=True,
            start_new_session=True,
            stdout=log_file,
            stderr=log_file,
            text=True
        )
        stdout, stderr = proc.wait()
        log_file.close()
        returncode = proc.returncode

        status = "SUCCESS" if returncode == 0 else "FAILED"
        return {
            "script": script_name,
            "status": status,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": returncode
        }

    except Exception as e:
        # 捕获 Popen 及 communicate 中的异常
        return {
            "script": script_name,
            "status": "ERROR",
            "stdout": "",
            "stderr": str(e),
            "returncode": None
        }

def run_scripts_parallel(scripts: List[str], max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    并行执行多个 shell 脚本。

    :param scripts: 脚本文件名列表
    :param max_workers: 最大并发线程数
    :return: 每个脚本的执行结果列表
    """
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_script = {
            executor.submit(run_shell_script, script): script
            for script in scripts
        }
        for future in as_completed(future_to_script):
            try:
                res = future.result()
            except Exception as e:
                # 捕获线程执行过程中未处理的异常
                res = {
                    "script": future_to_script[future],
                    "status": "ERROR",
                    "stdout": "",
                    "stderr": str(e),
                    "returncode": None
                }
            results.append(res)
    return results

if __name__ == "__main__":
    # 在这里指定要并行执行的脚本列表
    scripts_to_run = [
        "t_shell.sh",
        "t_shell.sh",
        # "t_shell.sh",
        # … 可以根据需要添加更多
    ]

    # 并行执行
    results = run_scripts_parallel(scripts_to_run, max_workers=4)

    # 输出汇总
    for r in results:
        print(f"脚本：{r['script']}  状态：{r['status']}")
        if r["stdout"]:
            print("── 标准输出 ──")
            print(r["stdout"])
        if r["stderr"]:
            print("── 标准错误 ──")
            print(r["stderr"])
        print(f"返回码：{r['returncode']}")
        print("=" * 40)
