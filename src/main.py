import torch,subprocess,requests,asyncio
from PIL import Image
from prompts.prompt_construct import PromptConstructor
from utils.data_tools import *
from utils.sample import ResponseSampler
from vwa.browser_env.auto_login import get_site_comb_from_filepath
from vwa.src.envs.browser import FastCachedwActionMatchingBrowserEnv
from vwa.src.evaluation import image_utils
def env_construct():
    """ 有一些参数没做处理 """

    # 建立环境
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    caption_image_fn = image_utils.get_captioning_fn(
        device, dtype, 'Salesforce/blip2-flan-t5-xl'
    )

    env = FastCachedwActionMatchingBrowserEnv(
        headless=True,  # 用无头模式
        slow_mo=0,
        action_set_tag='som',  # used by action caching
        observation_type='image_som',
        current_viewport_only=True,
        viewport_size={
            "width": 1280,
            "height": 2048,
        },
        save_trace_enabled=False,
        sleep_after_execution=2.5,
        # NOTE: captioning_fn here is used for LLM + captioning baselines.
        # This can be different from the captioning model used for evals.
        captioning_fn=caption_image_fn,
    )
    return env
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
    return config_file

async def asingle_task(task):
    """
    单个任务的执行
    """

    trajectory = []
    config_file = task_prepare(cache_dir, task)
    env = env_construct()  # 建立环境
    obs, info = await env.areset(options={"config_file": config_file})  # 环境重置

    for step in range(10): # max step num
        # 根据环境的obs建prompt
        pt_constructor = PromptConstructor(results_dir)
        messages = pt_constructor.construct_prompt(task,obs, info, trajectory, guidance='LIFT',examples='LIFT')
        # 根据prompt采样回答
        sampler = ResponseSampler(model_path="/data/wangzhenchuan/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct")
        response = sampler.sample(messages=messages,n=5)
        print(response)
        # 根据回答生成reward
        pass
        # reward最高的action作为env step
if __name__ == '__main__':

    # 参数
    cache_dir = '../cache'
    results_dir = '../results'

    # 获取数据,train_data是一个hf dataset
    train_task = dataset_construct()

    for epoch in range(10):
        idx = range(epoch*5, epoch*5+5) # 每次取5个数据
        sampled_task = train_task.select(list(idx)) # 从train_data中取出数据

        for _,task in enumerate(sampled_task):
            asyncio.run(asingle_task(task))
