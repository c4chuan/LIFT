import shlex
import subprocess
import os
import paramiko
from scp import SCPClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

def parallel_scp(
    remote_paths: List[str],
    local_dir: str = "/data/wangzhenchuan/Projects/LIFT/src/image_cache",
    ssh_key: str = "/data/wangzhenchuan/id_rsa_a100_wangzhenchuan",
    remote_user: str = "wangzhenchuan",
    remote_host: str = "192.168.1.5",
    max_workers: int = 4
) -> List[str]:
    """
    并行执行多条 scp 命令，将多条远程路径复制到本地对应路径。

    :param remote_paths: 远程服务器上的文件或目录列表
    :param local_dir: 本地目标路径列表（长度需与 remote_paths 一致）
    :param ssh_key: 私钥文件路径，例如 "id_rsa_user_8卡A6000"
    :param remote_user: 远程服务器用户名，例如 "user"
    :param remote_host: 远程服务器 IP 或主机名，例如 "192.168.1.6"
    :param max_workers: 最大并发线程数，默认为 5
    """
    # 1. 去重并保留顺序
    unique_remotes = list(dict.fromkeys(remote_paths))

    # 2. 为每个唯一远程路径计算本地目标路径，
    #    取倒数第二级目录名（如 '2'、'3'）+ '_' + 原始文件名
    local_paths = {}
    for src in unique_remotes:
        parts = src.rstrip('/').split('/')
        if len(parts) >= 2:
            parent = parts[-2]
        else:
            parent = "root"
        fname = parts[-1]
        local_name = f"{parent}_{fname}"
        local_paths[src] = os.path.join(local_dir, local_name)

    # 确保目录存在
    os.makedirs(local_dir, exist_ok=True)
    def scp_one(src: str, dst: str) -> Tuple[str, subprocess.CompletedProcess]:
        cmd = [
            "scp",
            "-i", ssh_key,
            "-r",
            f"{remote_user}@{remote_host}:{src}",
            dst
        ]
        # 调用 subprocess.run 执行 scp；capture_output=True 可捕获 stdout/stderr
        result = subprocess.run(cmd, capture_output=True, text=True)
        return src, result


    # 使用线程池并发执行 scp
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_src = {
            executor.submit(scp_one, src, dst): src
            for src, dst in local_paths.items()
        }

        # as_completed 按任务完成顺序迭代
        for future in as_completed(future_to_src):
            src = future_to_src[future]
            try:
                _, result = future.result()
                if result.returncode == 0:
                    print(f"[SUCCESS] {src} 已成功复制")  # 可替换为日志记录
                else:
                    print(f"[ERROR] {src} 复制失败，stderr: {result.stderr.strip()}")
            except Exception as exc:
                print(f"[EXCEPTION] {src} 复制时抛出异常: {exc}")
    # 4. 构造与原始列表一一对应的 local_paths（重复指向同一个本地文件）
    mapped_local_paths = [local_paths[src] for src in remote_paths]
    return mapped_local_paths


def create_ssh_client(remote_host: str, remote_user: str, ssh_key: str) -> paramiko.SSHClient:
    """
    创建并返回一个 SSH 客户端对象，建立到远程主机的连接。
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # 自动添加主机密钥
    ssh.load_system_host_keys()
    ssh.connect(remote_host, username=remote_user, key_filename=ssh_key)
    return ssh

def create_remote_dir(ssh_client: paramiko.SSHClient, remote_task_dir: str):
    """
    在远程服务器上创建 task_id 对应的目录，如果不存在的话。
    """
    remote_command = f"mkdir -p {remote_task_dir}"
    stdin, stdout, stderr = ssh_client.exec_command(remote_command)
    stderr_str = stderr.read().decode()
    if stderr_str:
        raise Exception(f"Error creating directory on remote server: {stderr_str}")
def scp_task(ssh_client: paramiko.SSHClient, task_id: int, image_path: str, remote_dir: str) -> str:
    """
    将本地文件复制到远程服务器，路径中包含任务 ID 作为文件夹。
    """
    remote_task_dir = os.path.join(remote_dir, str(task_id))  # 创建 task_id 对应的目录
    # 在远程服务器上创建目录
    create_remote_dir(ssh_client, remote_task_dir) # 确保远程目录存在

    file_name = os.path.basename(image_path)
    remote_file_path = os.path.join(remote_task_dir, file_name)

    with SCPClient(ssh_client.get_transport()) as scp:
        scp.put(image_path, remote_file_path)  # 将文件复制到远程路径

    return f"Successfully copied {image_path} to {remote_file_path}"


def parallel_scp_to_remote(
        task_ids: List[int],
        image_paths: List[List[str]],
        remote_dir: str = "/data/wangzhenchuan/Projects/LIFT/results",
        ssh_key: str = "/data/wangzhenchuan/id_rsa_a100_wangzhenchuan",
        remote_user: str = "wangzhenchuan",
        remote_host: str = "192.168.1.5",
        max_workers: int = 4
) -> List[str]:
    """
    并行将多条本地路径复制到远程服务器对应路径。
    """
    results = []

    # 创建 SSH 客户端
    ssh_client = create_ssh_client(remote_host, remote_user, ssh_key)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 为每个任务提交并行任务
        futures = []
        for task_id, paths in zip(task_ids, image_paths):
            for image_path in paths:
                futures.append(executor.submit(scp_task, ssh_client, task_id, image_path, remote_dir))

        # 获取所有任务的结果
        for future in futures:
            results.append(future.result())

    ssh_client.close()  # 关闭 SSH 连接
    return results


def run_command(cmd: str, timeout: int = 30) -> None:
    """
    执行一条 shell 命令，遇到错误抛出异常。
    """
    result = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed [{cmd}]: {result.stderr.strip()}")

def ensure_remote_dir(remote_host: str,
                      remote_user: str,
                      ssh_key: str,
                      remote_dir: str,
                      timeout: int = 30) -> None:
    """
    在远程机器上创建目录（-p），如果已存在则无视错误。
    """
    # 用 shlex.quote 对可变部分进行转义
    remote_dir_q = shlex.quote(remote_dir)
    ssh_cmd = (
        f"ssh -i {shlex.quote(ssh_key)} "
        f"{shlex.quote(remote_user)}@{shlex.quote(remote_host)} "
        f"mkdir -p {remote_dir_q}"
    )
    run_command(ssh_cmd, timeout=timeout)

def scp_one_file(remote_host: str,
                 remote_user: str,
                 ssh_key: str,
                 local_path: str,
                 remote_dir: str,
                 timeout: int = 60) -> str:
    """
    将单个 local_path 上传到 remote_user@remote_host:remote_dir。
    """
    # 确保本地文件存在
    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")

    # 先在远端创建目录
    ensure_remote_dir(remote_host, remote_user, ssh_key, remote_dir, timeout=timeout//2)

    # 构造 scp 命令
    local_q = shlex.quote(local_path)
    remote_target = f"{shlex.quote(remote_user)}@{shlex.quote(remote_host)}:{shlex.quote(remote_dir)}/"
    scp_cmd = f"scp -i {shlex.quote(ssh_key)} {local_q} {remote_target}"
    run_command(scp_cmd, timeout=timeout)

    basename = os.path.basename(local_path)
    return f"Copied {local_path} → {remote_user}@{remote_host}:{remote_dir}/{basename}"

def parallel_scp_to_remote_cmd_version(
    task_ids: List[int],
    image_paths: List[List[str]],
    remote_dir: str = "/data/wangzhenchuan/Projects/LIFT/results",
    ssh_key: str = "/data/wangzhenchuan/id_rsa_a100_wangzhenchuan",
    remote_user: str = "wangzhenchuan",
    remote_host: str = "192.168.1.5",
    max_workers: int = 4,
) -> List[str]:
    """
    并行将多组文件上传到远程服务器。

    :param task_ids:    与 image_paths 一一对应的任务 ID 列表
    :param image_paths: [[path1, path2, ...], [...], ...]
    :return:            每个上传操作的结果消息列表
    """
    if len(task_ids) != len(image_paths):
        raise ValueError("task_ids 和 image_paths 必须等长")

    results = []
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for task_id, paths in zip(task_ids, image_paths):
            for local_path in paths:
                # 构造该文件的远端子目录
                subdir = os.path.join(remote_dir, str(task_id))
                futures.append(
                    executor.submit(
                        scp_one_file,
                        remote_host,
                        remote_user,
                        ssh_key,
                        local_path,
                        subdir,
                    )
                )

        # 收集结果
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                # 你可以选择记录日志，或者将错误信息也加入 results
                results.append(f"ERROR: {e}")

    return results


# 示例调用
if __name__ == "__main__":
    remote_imgs = [
        "/data/wangzhenchuan/Projects/LIFT/results/0/intent.txt",
        "/data/wangzhenchuan/Projects/LIFT/results/2/step_0_obs.png",
        "/data/wangzhenchuan/Projects/LIFT/results/2/step_0_obs.png",
        "/data/wangzhenchuan/Projects/LIFT/results/3/step_0_obs.png",
    ]
    local_dir = "/data/wangzhenchuan/Projects/LIFT/src/image_cache"
    paths = parallel_scp(
        remote_paths=remote_imgs,
        local_dir = local_dir,
        ssh_key="/data/wangzhenchuan/id_rsa_a100_wangzhenchuan",
        remote_user="wangzhenchuan",
        remote_host="192.168.1.5",
        max_workers=4
    )
    print(paths)
