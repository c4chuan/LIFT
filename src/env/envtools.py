import os,json,subprocess,requests
from PIL import Image
from vwa.browser_env.auto_login import get_site_comb_from_filepath
from vwa.browser_env.env_config import (
    CLASSIFIEDS,
    CLASSIFIEDS_RESET_TOKEN,
    GITLAB,
    HOMEPAGE,
    MAP,
    REDDIT,
    SHOPPING,
    SHOPPING_ADMIN,
    WIKIPEDIA,
)
ENV_URLS_SH = f"""
export CLASSIFIEDS="{CLASSIFIEDS}"
export CLASSIFIEDS_RESET_TOKEN="{CLASSIFIEDS_RESET_TOKEN}"
export SHOPPING="{SHOPPING}"
export REDDIT="{REDDIT}"
export WIKIPEDIA="{WIKIPEDIA}"
export SHOPPING_ADMIN="{SHOPPING_ADMIN}"
export GITLAB="{GITLAB}"
export MAP="{MAP}"
export HOMEPAGE="{HOMEPAGE}"
""".strip()

def task_prepare(cache_dir,task):
    """
    给定task，进行auto_login以及load images（如果有的话）
    """

    # 把任务的config存到cache_dir下
    config_file = os.path.join(cache_dir, f"{task['task_id']}.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(task, fp=f, indent=4,sort_keys=True)

    # Load task.
    print("###Load Task###")
    with open(config_file) as f:
        _c = json.load(f)
        intent = _c["intent"]
        task_id = _c["task_id"]
        image_paths = _c.get("image", None)
        images = []

        # automatically login
        print("###Auto Login###")
        if _c["storage_state"]:
            cookie_file_name = os.path.basename(_c["storage_state"])
            print(cookie_file_name)
            comb = get_site_comb_from_filepath(cookie_file_name)
            print(comb)
            temp_dir = "../cache"
            print(temp_dir)
            # subprocess to renew the cookie
            print("run sub")
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
            print(_c["storage_state"])
            assert os.path.exists(_c["storage_state"])
            # update the config file
            print("update")
            config_file = f"{temp_dir}/{os.path.basename(config_file)}"
            with open(config_file, "w") as f:
                json.dump(_c, f)
            print("dump json")

        # Load input images for the task, if any.
        print("###Load Images###")
        if image_paths is not None:
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
            print("No input images.")

    task_info = {
        "config_file": config_file,
        "task_id": task_id,
        "intent": intent,
        "images": images,
    }
    print(f"Intent: {intent}")
    return task_info

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

    python -m browser_env.auto_login
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