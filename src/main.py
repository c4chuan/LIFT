import os.path

import cv2
import torch,subprocess,requests,asyncio,pickle
from PIL import Image
from prompts.prompt_construct import PromptConstructor
from utils.data_tools import *
from utils.sample import ResponseSampler
from vwa.browser_env.auto_login import get_site_comb_from_filepath
from vwa.src.envs.browser import FastCachedwActionMatchingBrowserEnv
from vwa.src.evaluation import image_utils
from src.reward.rewarder import Rewarder
def env_construct():
    """ 有一些参数没做处理 """

    # 建立环境
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    env = FastCachedwActionMatchingBrowserEnv(
        headless=True,  # 用无头模式
        slow_mo=0,
        action_set_tag='som',  # used by action caching
        observation_type='image_som_without_caption',
        current_viewport_only=True,
        viewport_size={
            "width": 1280,
            "height": 2048,
        },
        save_trace_enabled=False,
        sleep_after_execution=2.5,
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

    # 初始化Rewarder
    rewarder = Rewarder("/data/wangzhenchuan/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct")
    for step in range(1): # max step num
        # 根据环境的obs建prompt
        pt_constructor = PromptConstructor(results_dir)
        messages = pt_constructor.construct_prompt(task,obs, info, trajectory, guidance='LIFT',examples='LIFT')
        # 以pickle的形式保存messages
        with open(f"../data/messages.pkl", "wb") as f:
            pickle.dump(messages, f)
        # 根据prompt采样回答
        num_sample = 5
        # sampler = ResponseSampler(model_path="/data/wangzhenchuan/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct")
        # response = sampler.sample(messages=messages,n=num_sample)
        payload = {
            "messages": messages,
            "n": 5  # 想要的样本数量，可改
        }
        response = requests.post(url = "http://localhost:7451/sample", data=json.dumps(payload))
        response_texts = response.json()['samples']
        print(response)
        # 根据回答生成reward
        reward = -1
        action_index = -1
        for index in range(num_sample):
            response_text = response_texts[index]
            save_dir = f'../results/response{index}'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            image_path = save_dir+'/o_image.png'
            cv2.imwrite(image_path,cv2.cvtColor(obs['image'],cv2.COLOR_BGR2RGB))
            shift_reward,zoom_reward = rewarder.reward(response_text,image_path,visualize=True,visual_save=save_dir)
            with open(save_dir+'/response.txt','w',encoding='utf-8') as f:
                f.write(response_text)
            with open(save_dir+'/message.txt','w',encoding='utf-8') as f:
                f.write(messages[-1]['content'][0]['text'])
            print(f'response{index}的shift_reward是{shift_reward},zoom_reward是{zoom_reward}')
            cur_reward = shift_reward+zoom_reward
            if cur_reward > reward:
                reward = cur_reward
                action_index = index
        # reward最高的action作为env step
        action_text = response_texts[action_index]

if __name__ == '__main__':

    # 参数
    cache_dir = '../cache'
    results_dir = '../results'

    # 获取数据,train_data是一个hf dataset
    train_task = dataset_construct()

    num_task =1
    for epoch in range(1):
        idx = range(epoch*num_task, epoch*num_task+num_task) # 每次取5个数据
        sampled_task = train_task.select(list(idx)) # 从train_data中取出数据

        for _,task in enumerate(sampled_task):
            asyncio.run(asingle_task(task))
