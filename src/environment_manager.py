import asyncio,os,re

import imageio
from box import Box
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from fastapi import FastAPI, BackgroundTasks
from contextlib import asynccontextmanager
from uvicorn import run as uvicorn_run
from pydantic import BaseModel

from src.agentic.policy import EnvLIFTConstructor
from src.utils.llm_config import construct_llm_config
from src.utils.data_tools import dataset_construct
from src.prompts.prompt_construct import PromptConstructor
from src.env.envtools import *
from visualwebarena.src.llms.tokenizer import Tokenizer
from vwa.src.envs.browser import FastCachedwActionMatchingBrowserEnv
from vwa.src.envs.actions import create_none_action,ActionTypes,create_id_based_action
from vwa.src.helper_functions import *
# from vwa.src.evaluation.vwa_evaluators import evaluator_router
from src.utils.scp_tools import parallel_scp_to_remote,parallel_scp_to_remote_cmd_version
from src.reward.reward_tools import get_action_id_answer,action_format_reward
from difflib import SequenceMatcher
from visualwebarena.src.agentic.policy import MCoTPolicyPConstructor

class ResponseList(BaseModel):
    responses: List[str]
@dataclass
class VWATask:
    task_id: int
    steps: int
    trajectory: list
    state_trajectory: list
    action_history: list
    env: FastCachedwActionMatchingBrowserEnv
    task_info: Dict[str, Any]
    config_file: Dict[str, Any]
    obs_info: Dict[str, Any]

class EnvironmentManager:
    def __init__(self, initial_configs,tasks):
        self.configs = initial_configs
        self.tasks = tasks
        self.task_pointer = 0
        self.cache_dir = self.configs.cache_dir
        self.results_dir = self.configs.results_dir
        # Free environment pool
        self.idle_tasks: List[VWATask] = []
        # Busy environments waiting for step
        self.busy_tasks: List[VWATask] = []
        # Message queue (ready for consumption by model)
        self.message_queue: List[Tuple[VWATask, Dict[str, Any]]] = []
        self.prev_tasks = []
        self.env_name =  initial_configs.env_name
        self.max_task_steps = initial_configs.max_task_steps
        self.pct = EnvLIFTConstructor(initial_configs.instruction_path,save_dir=self.results_dir, lm_config=construct_llm_config(), tokenizer=Tokenizer(provider="openai", model_name="gpt-4o"))
        # self.pct = PromptConstructor(self.results_dir)
        if initial_configs.initial_refresh_env:
            self.refresh_env(initial_configs.env_name)

    async def initial_message_construct(self,max_num_envs):
        """创建max_num_envs个空环境实例，并建立初始化消息分别push进task_queue和message_queue"""
        for i in range(max_num_envs):
            self.busy_tasks.append(self.create_new_task())
        await self.parallel_produce(self.busy_tasks,[create_none_action() for _ in range(len(self.busy_tasks))])

    def find_index(self,task_list,task):
        for i,_task in enumerate(task_list):
            if _task.task_id == task.task_id:
                return i

    def is_new_task_needed(self,task,action):
        if action.action_type == ActionTypes.NONE or action.action_type == ActionTypes.STOP or task.steps >= self.max_task_steps:
            return 1
        else:
            return 0

    def create_new_task(self):
        if self.task_pointer < len(self.tasks):
            new_task = VWATask(
                task_id = self.tasks[self.task_pointer]['task_id'],
                steps=0,
                trajectory=[],
                state_trajectory=[],
                action_history=[],
                env=FastCachedwActionMatchingBrowserEnv(
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
                ),
                task_info = {},
                config_file = self.tasks[self.task_pointer],
                obs_info = {}
            )
            self.task_pointer += 1
            self.task_pointer = self.task_pointer % 200
            return new_task

    def get_task_flag_list(self,tasks,actions):
        task_flag_list = []
        for task,action in zip(tasks,actions):
            if self.is_new_task_needed(task,action):
                task_flag_list.append(True)
            else:
                task_flag_list.append(False)
        return task_flag_list


    async def parallel_produce(self,Tasks,Actions):
        """并行生产messages"""
        coros = []
        config_files = [] # for env.reset
        counter = 0
        flag_list = self.get_task_flag_list(Tasks,Actions)

        for task,action,flag in zip(Tasks,Actions,flag_list):
            if flag:
                if task in self.prev_tasks:
                    config_files.append(self.tasks[self.task_pointer+counter])
                    counter += 1
                else:
                    config_files.append(task.config_file)
        if len(config_files) > 0:
            task_infos = parallel_prepare(self.cache_dir,config_files,max_workers=8)
            info_index = 0

        processed_tasks = []
        for task,action,flag in zip(Tasks,Actions,flag_list):
            if flag:
                if task in self.prev_tasks:
                    new_task = self.create_new_task()
                    self.prev_tasks[self.find_index(self.prev_tasks,task)] = new_task
                    if task in self.busy_tasks:
                        self.busy_tasks[self.find_index(self.busy_tasks, task)] = new_task
                    new_task.task_info = task_infos[info_index]
                    coros.append(new_task.env.areset(options={"config_file":task_infos[info_index]['config_file']}))
                    processed_tasks.append(new_task)
                else:
                    task.task_info = task_infos[info_index]
                    coros.append(task.env.areset(options={"config_file":task_infos[info_index]['config_file']}))
                    processed_tasks.append(task)
                info_index += 1
            else:
                coros.append(task.env.astep(action))
                processed_tasks.append(task)

        # asyncio.gather 会并发调度所有 coroutine
        results = await asyncio.gather(*coros, return_exceptions=False)
        # 构造messages
        messages = []
        task_ids = []

        for task,action,result,flag in zip(processed_tasks,Actions,results,flag_list):
            if flag:
                obs,info = result
                state_info = {"observation": obs, "info": info, "url": task.env.page.url}
                task.state_trajectory.append(state_info)
                # message = self.pct.construct_messages(task,obs,info,guidance='LIFT',examples='LIFT')
                message = self.pct.construct(
                                            task_id = task.task_id,
                                            trajectory = task.state_trajectory,
                                             intent = task.task_info['intent'],
                                             page_screenshot_img = Image.fromarray(obs['image']),
                                             images = task.task_info['images'],
                                             meta_data = {"action_history": ["None"]},)
                task.obs_info = info
            else:
                obs,action_reward,_,_,info = result # None,0.0,
                if obs['image'] is None:
                    obs = task.state_trajectory[-1]['observation']
                    action_reward = 0.0
                    info = task.state_trajectory[-1]['info']
                state_info = {"observation": obs, "info": info, "url": task.env.page.url}
                task.state_trajectory.append(action)
                task.state_trajectory.append(state_info)
                action_str = get_action_description(
                    action,
                    state_info["info"]["observation_metadata"],
                    action_set_tag="som",
                    prompt_constructor=self.pct
                )
                task.action_history.append(action_str)


                # message = self.pct.construct_messages(task,obs,info,guidance='LIFT',examples='LIFT')
                message = self.pct.construct(
                                            task_id=task.task_id,
                                            trajectory=task.state_trajectory,
                                             intent=task.task_info['intent'],
                                             page_screenshot_img=Image.fromarray(obs['image']),
                                             images=task.task_info['images'],
                                             meta_data={"action_history": task.action_history}, )
                task.obs_info = info
            messages.append(message)
            task_ids.append(task.task_id)


            self.send_images_to_server(task_ids,messages)

            # 生成完毕以后加入idle_tasks
            self.idle_tasks.append(task)
            self.busy_tasks.remove(task)
            print(f"Task:{task.task_id}-Action:{str(action.action_type)}step完毕-加入消息队列")

        self.message_queue += [(task, message) for task, message in zip(processed_tasks, messages)]
        print(
            f"现在处于空闲状态的任务有{len(self.idle_tasks)}个，task_id分别是{str([t.task_id for t in self.idle_tasks])}")
        print(
            f"现在处于生产状态的任务有{len(self.busy_tasks)}个，task_id分别是{str([t.task_id for t in self.busy_tasks])}")
        print(
            f"现在处于消息队列中的任务有{len(self.message_queue)}个，task_id分别是{str([t.task_id for t, _ in self.message_queue])}")

        for task in processed_tasks:
            if task in self.prev_tasks:
                self.prev_tasks.remove(task) # 生产完毕，可以清除

    def get_valid_action_rewards(self,responses):
        """给每一个action计算是否可以进行迭代"""
        print(len(responses))
        print(
            f"现在处于prev_tasks队列中的任务有{len(self.prev_tasks)}个，task_id分别是{str([t.task_id for t, _ in self.message_queue])}")
        valid_action_rewards = []
        batch_size = int(len(responses) / len(self.prev_tasks))
        responses_list = [responses[i:i + batch_size] for i in range(0, len(responses), batch_size)]
        assert len(responses_list) == len(self.prev_tasks) #  responses_list的每个元素对应每个环境去验证的responses
        for responses,task in zip(responses_list,self.prev_tasks):
            for response in responses:
                if action_format_reward(response) == 0: # 如果格式不对，奖励为0
                    valid_action_rewards.append(0.0)
                else: # 如果格式正确，则进行验证
                    action_info = get_action_id_answer(response)
                    var = self.get_reward_by_action_info(task,action_info,response)
                    valid_action_rewards.append(var)

        return valid_action_rewards

    def get_reward_by_action_info(self,task,action_info,response):
        element_id = action_info['element_id']
        answer = action_info['answer']
        url = action_info['url']
        if element_id == -1:  # 说明是一个与element_id无关的action
            return 0.0
        elif int(element_id) > 0:
            var = self.evaluate_action_validation(task, element_id=element_id)  # 判断element_id是否超出环境范围，错了扣分，正确不加分
            var += self.evaluate_action_click_alignment(task, element_id, response)
            return  var
        elif isinstance(answer, str):  # 说明是stop类型动作
            var = self.evaluate_action_validation(task, answer=answer)  # 错了不扣分，正确加分
            return var
        elif isinstance(url, str):  # 说明是click类型动作
            if "http://" in url or "https://" in url:
                return 0.0
            else:
                return -1.0
        else:
            return 0

    def extract_summary(self,response):
        pattern = r'<summary>([\s\S]*?)</summary>'
        match = re.search(pattern, response)
        if match:
            summary = match.group(1)
            return summary
        else:
            return ""

    def get_element_id_description(self,string):
        # 找到所有 [...] 的内容
        matches = re.findall(r'\[([^\]]*)\]', string)
        # 如果有找到，就返回最后一个，否则返回空
        return matches[-1] if matches else ''
    def evaluate_action_click_alignment(self,task,element_id,response):
        # 提取response 中的summary
        summary = self.extract_summary(response)
        if summary != "" and element_id in task.obs_info['observation_metadata']['image']['obs_nodes_info']:
            id_description = self.get_element_id_description(task.obs_info['observation_metadata']['image']['obs_nodes_semantic_info'][element_id])
            for component in id_description.split(" "):
                if component in summary:
                    return 0.5
            return 0.0
        else:
            return 0.0

    def evaluate_action_validation(self,task,element_id=None,answer=None):
        """验证action的有效性以及stop action的成功率"""
        if element_id is not None:
            if element_id in task.obs_info['observation_metadata']['image']['obs_nodes_info']: # 判断element_id是否超出环境范围
                return 0.0
            else:
                return -1.0
        elif answer is not None:
            # 太烦人了这个evaluator，import都要import半天拖慢速度，后面再说
            # evaluator = evaluator_router(
            #     task.config_file
            # )
            # score = await evaluator(
            #     trajectory=task.state_trajectory,
            #     config_file=task.config_file,
            #     page=task.env.page
            # )
            # return score
            return 0.0
        else:
            return 0.0




    async def feed_responses(self, responses):
        """
        Consumer calls this to return responses to the environment manager.
        """
        actions = []
        # 先检查 busy_tasks 中是否有任务，有的话，就先处理（也就是等它们跑完），再进行后续 feed
        while len(self.busy_tasks) > 0:
            await asyncio.sleep(0.1)


        for response in responses:
            action_str = self.pct.extract_action(response)
            if action_str == None or action_str == "None":
                action = create_none_action()
                action.update({"raw_prediction": response})
                actions.append(action)
            else:
                action = create_id_based_action(action_str)
                action.update({"raw_prediction": response})
                actions.append(action)

        tasks_to_produce = []
        for task,response,action in zip(self.prev_tasks,responses,actions):
            task.steps += 1
            prev = self.pct.exstract_summary(response)
            task.trajectory.append(prev)
            self.idle_tasks.remove(task)
            self.busy_tasks.append(task)
            tasks_to_produce.append(task)
        # Trigger production without awaiting
        asyncio.create_task(self.parallel_produce(tasks_to_produce, actions))




    def get_messages(self, num):
        """
        Consumer calls this to retrieve up to `num` messages for sampling.
        """
        print(f"需要取出前{num}条消息")
        return_messages = []
        try:
            for task,message in self.message_queue[:num]:
                print(f"取出Task{task.task_id}的消息")
                return_messages.append(message)
                self.prev_tasks.append(task)
            self.message_queue = self.message_queue[num:]
            print(
                f"现在处于消息队列中的任务有{len(self.message_queue)}个，task_id分别是{str([t.task_id for t, _ in self.message_queue])}")
        except  IndexError:
            if len(self.busy_tasks) > 0:
                print(f"现在的消息队列中的有效消息不足，正在生产中")
                return []
            elif self.task_pointer >= len(self.tasks):
                print(f"任务队列已用完")
                return "Finished"
            else:
                print(f"未知情况")
        return return_messages

    def collect_images_paths(self,task_messages):
        """
        每条messages中的所有用户图片
        """
        image_paths = []
        for task_msg in task_messages:
            msg_image_paths = []
            for msg in task_msg:
                if msg['role'] == 'user':
                    for content in msg['content']:
                        if 'image' in content:
                            msg_image_paths.append(content['image'])
            image_paths.append(msg_image_paths)
        return image_paths

    def send_images_to_server(self,task_ids,messages):
        """
        Consumer calls this to send messages to the server.
        """
        image_paths = self.collect_images_paths(messages)
        if self.configs.scp_version == "client":
            parallel_scp_to_remote(task_ids, image_paths)
        elif self.configs.scp_version == "cmd":
            parallel_scp_to_remote_cmd_version(task_ids, image_paths)



    def get_val_messages(self, num):
        """
        Consumer calls this to retrieve up to `num` messages for sampling.
        """
        print(f"需要取出前{num}条消息")
        return_messages = []
        try:
            for task,message in self.message_queue[:num]:
                print(f"取出Task{task.task_id}的消息")
                return_messages.append(message)
            print(
                f"现在处于消息队列中的任务有{len(self.message_queue)}个，task_id分别是{str([t.task_id for t, _ in self.message_queue])}")
        except  IndexError:
            if len(self.busy_tasks) > 0:
                print(f"现在的消息队列中的有效消息不足，正在生产中")
                return []
            elif self.task_pointer >= len(self.tasks):
                print(f"任务队列已用完")
                return "Finished"
            else:
                print(f"未知情况")
        return return_messages

    def refresh_env(self,env_name):
        """
        刷新环境
        """
        refresh_env_login()
        reset_env(env_name)

# Instantiate manager
dataset = dataset_construct()
tasks_cfg = list(dataset)
initial_configs = Box({
    'max_num_envs': 8,
    'initial_refresh_env': False,
    'cache_dir': './.auth',
    'env_name': 'classifields',
    'results_dir': '/data/wangzhenchuan/Projects/LIFT/results',
    'max_task_steps': 4,
    'scp_version': 'cmd', # 'client' or 'cmd'
    'type': 'remote',
    'target_server': '192.168.1.5',
    'instruction_path': '/data/wangzhenchuan/Projects/LIFT/visualwebarena/src/prompts/vwa/jsons/lift.json'
})
env_manager = EnvironmentManager(initial_configs, tasks_cfg)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 服务器跑起来以后，再使用当前事件循环安全地并行 reset 所有环境
    asyncio.create_task(
        env_manager.initial_message_construct(env_manager.configs.max_num_envs)
    )
    yield  # 控制权交给 FastAPI，让它启动并开始接收请求

app = FastAPI(lifespan=lifespan)

@app.get('/get_length')
async def api_get_length():
    return {"length": len(env_manager.tasks)}

@app.get('/get_messages')
async def api_get_messages(num: int):
    # Wait until enough messages are available or no busy tasks remain
    while env_manager.busy_tasks:
        await asyncio.sleep(0.1)
    msgs = env_manager.get_messages(min(num, len(env_manager.message_queue)))
    return {'messages': msgs}

@app.get('/get_val_messages')
async def api_get_val_messages(num: int):
    # Wait until enough messages are available or no busy tasks remain
    while len(env_manager.message_queue) < num and env_manager.busy_tasks:
        await asyncio.sleep(0.1)
    msgs = env_manager.get_val_messages(min(num, len(env_manager.message_queue)))
    return {'messages': msgs}

@app.post('/feed_responses')
def api_feed_responses(responses: List[str], background_tasks: BackgroundTasks):
    background_tasks.add_task(env_manager.feed_responses, responses)
    return {'status': 'accepted'}

@app.get('/refresh_env')
def api_refresh_env():
    env_manager.refresh_env(env_manager.env_name)
    return {'status': 'refreshed'}

@app.get('/get_is_last_batch')
def api_get_is_last_batch():
    return {'flag': env_manager.task_pointer>len(env_manager.tasks)}

@app.post('/get_valid_action_rewards')
def api_get_valid_action_rewards(responselist: ResponseList):
    rewards = env_manager.get_valid_action_rewards(responselist.responses)
    return {'rewards': rewards}

if __name__ == '__main__':
    uvicorn_run(app, host='0.0.0.0', port=7333)
# if __name__ == '__main__':
#     dataset = dataset_construct()
#     tasks = [task for _,task in enumerate(dataset)]
#     initial_configs = {
#         "max_num_envs": 4,
#         "initial_refresh_env": True,
#         "cache_dir": "../cache",
#         "env_name": "classifields",
#         "results_dir": "../results",
#     }
#     initial_configs = Box(initial_configs)
#     env_manager = EnvironmentManager(initial_configs,tasks)
#
#     # 取数据测试
#     for _ in range(5):
#         batch = env_manager.get_messages(2)
#         responses = []
#         for message in batch:
#             payload = {
#                 "messages": message,
#                 "n": 1  # 想要的样本数量，可改
#             }
#             response = requests.post(url="http://localhost:7451/sample", data=json.dumps(payload))
#             responses += response.json()['samples']
#         asyncio.run(env_manager.feed_responses(responses))





