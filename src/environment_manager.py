import asyncio,os
from box import Box
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from fastapi import FastAPI, BackgroundTasks
from contextlib import asynccontextmanager
from uvicorn import run as uvicorn_run
from src.utils.data_tools import dataset_construct
from src.prompts.prompt_construct import PromptConstructor
from src.env.envtools import *
from vwa.src.envs.browser import FastCachedwActionMatchingBrowserEnv
from vwa.src.envs.actions import create_none_action,ActionTypes,create_id_based_action


@dataclass
class VWATask:
    task_id: int
    steps: int
    trajectory: list
    env: FastCachedwActionMatchingBrowserEnv
    task_info: Dict[str, Any]
    config_file: Dict[str, Any]

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

    def create_new_task(self):
        if self.task_pointer < len(self.tasks):
            new_task = VWATask(
                task_id = self.tasks[self.task_pointer]['task_id'],
                steps=0,
                trajectory=[],
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
                config_file = self.tasks[self.task_pointer]
            )
            self.task_pointer += 1
            return new_task

    async def parallel_produce(self,Tasks,Actions):
        """并行生产messages"""
        coros = []
        config_files = [] # for env.reset
        counter = 0
        for task,action in zip(Tasks,Actions):
            if action.action_type == ActionTypes.NONE or action.action_type == ActionTypes.STOP:
                if task in self.prev_tasks:
                    config_files.append(self.tasks[self.task_pointer+counter])
                    counter += 1
                else:
                    config_files.append(task.config_file)
        if len(config_files) > 0:
            task_infos = parallel_prepare(self.cache_dir,config_files,max_workers=8)
            info_index = 0

        processed_tasks = []
        for task,action in zip(Tasks,Actions):
            if action.action_type == ActionTypes.NONE or action.action_type == ActionTypes.STOP:
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
        pct = PromptConstructor(self.results_dir)
        messages = []
        for task,action,result in zip(processed_tasks,Actions,results):
            if action.action_type == ActionTypes.NONE or action.action_type == ActionTypes.STOP:
                obs,info = result
                message = pct.construct_messages(task,obs,info,guidance='LIFT',examples='LIFT')
            else:
                obs,_,_,_,info = result
                message = pct.construct_messages(task,obs,info,guidance='LIFT',examples='LIFT')
            messages.append(message)

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

    async def feed_responses(self, responses):
        """
        Consumer calls this to return responses to the environment manager.
        """
        actions = []
        pct = PromptConstructor(self.results_dir)
        for response in responses:
            action_str = pct.extract_action(response)
            if action_str == None:
                actions.append(create_none_action())
            else:
                actions.append(create_id_based_action(action_str))

        tasks_to_produce = []
        for task,response,action in zip(self.prev_tasks,responses,actions):
            task.steps += 1
            prev = pct.exstract_summary(response)
            task.trajectory.append(prev)
            self.idle_tasks.remove(task)
            self.busy_tasks.append(task)
            tasks_to_produce.append(task)
        # Trigger production without awaiting
        asyncio.create_task(self.parallel_produce(tasks_to_produce, actions))

    async def feed_actions(self, actions):
        """
        Consumer calls this to return actions to the environment manager.
        """
        responses = ["" for _ in range(len(actions))]
        tasks_to_produce = []
        pct = PromptConstructor(self.results_dir)
        for task,response,action in zip(self.prev_tasks,responses,actions):
            task.steps += 1
            prev = pct.exstract_summary(response)
            task.trajectory.append(prev)
            self.idle_tasks.remove(task)
            self.busy_tasks.append(task)
            tasks_to_produce.append(task)
            # Trigger production without awaiting
            asyncio.get_event_loop().create_task(self.parallel_produce(tasks_to_produce, actions))
            self.prev_tasks.clear()

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
    'max_num_envs': 4,
    'initial_refresh_env': True,
    'cache_dir': './.auth',
    'env_name': 'classifields',
    'results_dir': '../results'
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

@app.get('/get_messages')
async def api_get_messages(num: int):
    # Wait until enough messages are available or no busy tasks remain
    while len(env_manager.message_queue) < num and env_manager.busy_tasks:
        await asyncio.sleep(0.1)
    msgs = env_manager.get_messages(min(num, len(env_manager.message_queue)))
    return {'messages': msgs}

@app.post('/feed_responses')
def api_feed_responses(responses: List[str], background_tasks: BackgroundTasks):
    background_tasks.add_task(env_manager.feed_responses, responses)
    return {'status': 'accepted'}

@app.get('/refresh_env')
def api_refresh_env():
    env_manager.refresh_env(env_manager.env_name)
    return {'status': 'refreshed'}

@app.post('/feed_actions')
def api_feed_actions(actions: List[Any], background_tasks: BackgroundTasks):
    background_tasks.add_task(env_manager.feed_actions, actions)
    return {'status': 'accepted'}

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





