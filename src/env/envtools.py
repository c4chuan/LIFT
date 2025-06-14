import os,json,subprocess,requests
import argparse
import time

from PIL import Image
from vwa.runners.utils.prepare_vwa import main as prepare_vwa_main
from concurrent.futures import ThreadPoolExecutor, as_completed
from vwa.browser_env.auto_login import get_site_comb_from_filepath
from vwa.browser_env.env_config import (
    CLASSIFIEDS,
    CLASSIFIEDS_RESET_TOKEN,
    # GITLAB,
    HOMEPAGE,
    # MAP,
    REDDIT,
    SHOPPING,
    # SHOPPING_ADMIN,
    WIKIPEDIA,
)
ENV_URLS_SH = f"""
export CLASSIFIEDS="{CLASSIFIEDS}"
export CLASSIFIEDS_RESET_TOKEN="{CLASSIFIEDS_RESET_TOKEN}"
export SHOPPING="{SHOPPING}"
export REDDIT="{REDDIT}"
export WIKIPEDIA="{WIKIPEDIA}"
export HOMEPAGE="{HOMEPAGE}"
""".strip()

def task_prepare(cache_dir,task):
    """
    给定task，进行auto_login以及load images（如果有的话）
    """

    # 把任务的config存到cache_dir下
    config_file = os.path.join(cache_dir, f"{task['task_id']}.json")
    # 尝试写入 JSON，并捕获可能的序列化错误
    try:
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(task, f, indent=4, sort_keys=True)
    except (TypeError, ValueError) as e:
        # 序列化失败时，打印错误并抛出或处理
        print(f"[ERROR] 将 task 序列化为 JSON 时出错: {e!r}")

    print("Loading config from:", config_file)
    # Load task.
    with open(config_file) as f:
        content = f.read()
        _c = json.loads(content)
        intent = _c["intent"]
        task_id = _c["task_id"]
        image_paths = _c.get("image", None)
        images = []

        # automatically login
        if _c["storage_state"]:
            cookie_file_name = os.path.basename(_c["storage_state"])
            comb = get_site_comb_from_filepath(cookie_file_name)
            temp_dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
            # 同理对 temp_dir
            os.makedirs(temp_dir, exist_ok=True)
            subprocess.run(
                [
                    "python",
                    "-m",
                    "vwa.browser_env.auto_login",
                    "--auth_folder",
                    temp_dir,
                    "--site_list",
                    *comb,
                ]
            )
            _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
            assert os.path.exists(_c["storage_state"])
            # update the config file
            config_file = f"{temp_dir}/{os.path.basename(config_file)}"


        if image_paths is not None:
            # Load input images for the task, if any.
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            for image_path in image_paths:
                # Load image either from the web or from a local path.
                if image_path.startswith("http"):
                    req = requests.get(
                        image_path,
                        headers={"User-Agent": "Mozilla/5.0"},
                        stream=True
                    )
                    input_image = Image.open(req.raw)
                else:
                    input_image = Image.open(image_path)

                images.append(input_image)
        else:
            print("No input images found.")

    task_info = {
        "config_file": config_file,
        "task_id": task_id,
        "intent": intent,
        "images": images,
    }
    print(f"Intent: {intent}")
    return task_info

def parallel_prepare(cache_dir, all_tasks, max_workers=4):
    print("Starting parallel prepare.")
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        # map 会按 all_tasks 的顺序返回结果
        results = list(exe.map(lambda t: task_prepare(cache_dir, t), all_tasks))
    print("Finish parallel prepare.")
    return results

def refresh_env_login():
    print("Refreshing login tokens.")
    os.makedirs("./.auth", exist_ok=True)

    dataset = os.environ.get("DATASET")
    assert dataset in ["webarena", "visualwebarena"], f"Unknown dataset {dataset=}"
    if dataset == "visualwebarena":
        urls = f"""
        export DATASET=visualwebarena
        {ENV_URLS_SH}
        """.replace(" "*4, "").strip()
    else:
        urls = f"""
        export DATASET=webarena
        {ENV_URLS_SH}
        """.replace(" "*4, "").strip()

    login_script_content = f"""
    {urls}

    python -m vwa.browser_env.auto_login
    """.replace(" "* 4, "").strip()

    login_script_path = "./.auth/refresh_login.sh"
    with open(login_script_path, 'w', encoding='utf-8') as fwrite:
        fwrite.write(login_script_content)

    process = subprocess.Popen(
        f"sh {login_script_path}",
        shell=True,
        start_new_session=True,
        text=True
    )
    process.wait()

    print("Done refreshing login tokens.")
    return

def reset_env(env_name: str):
    # reset the environment
    arg = argparse.Namespace(
        mode="reset",
        env=env_name,
        force=True
    )
    prepare_vwa_main(arg)
    print(f"Done resetting the environment {env_name}.")
    return

def reserve_env(env_name: str):
    # reserve the environment
    print(f"Reserving the environment {env_name}.")
    arg = argparse.Namespace(
        mode="reserve",
        env=env_name
    )
    prepare_vwa_main(arg)
    print(f"Done reserving the environment {env_name}.")
    return


def free_env(env_name: str):
    # free the environment
    print(f"Freeing the environment {env_name}.")
    arg = argparse.Namespace(
        mode="free",
        env=env_name
    )
    prepare_vwa_main(arg)
    print(f"Done freeing the environment {env_name}.")
    return