from src.utils.data_tools import dataset_construct
from vwa.src.envs.browser import FastCachedwActionMatchingBrowserEnv,Action
from src.main import task_prepare
from prompts.prompt_construct import PromptConstructor
from vwa.src.envs.actions import create_id_based_action
from playwright.sync_api import sync_playwright
import asyncio
import requests
async def main():
    dataset = dataset_construct()
    config_file = task_prepare('../cache',  dataset.select([0])[0])
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
    obs,info = await env.areset(options={"config_file": config_file})
    with open('../results/response3/response.txt','r',encoding='utf-8') as f:
        response = f.read()
    pt = PromptConstructor('../results')
    action_str = pt.extract_action(response)
    print(action_str)
    action = create_id_based_action(action_str)
    print(action)
    obs,_,_,_,info = await env.astep(action)
    pass
if __name__ == '__main__':
    asyncio.run(main())


    # with open('../results/response3/response.txt','r',encoding='utf-8') as f:
    #     response = f.read()
    # pt = PromptConstructor('../results')
    # action_str = pt.extract_action(response)
    # print(action_str)
    # action = create_id_based_action(action_str)
    # print(action)

    # with sync_playwright() as p:
    #     browser = p.chromium.launch(headless=True)  # 启动浏览器
    #     page = browser.new_page()  # 打开新的页面
    #     page.goto('http://192.168.1.12:9980/index.php?page=login')  # 访问网页
    #     print(page.title())  # 输出页面标题
    #     browser.close()


