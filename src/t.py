from src.utils.data_tools import dataset_construct
from vwa.src.envs.browser import FastCachedwActionMatchingBrowserEnv,Action
from src.main import task_prepare
from prompts.prompt_construct import PromptConstructor
from vwa.src.envs.actions import create_id_based_action
from playwright.sync_api import sync_playwright

def main():
    dataset = dataset_construct()
    config_file = task_prepare('../cache', dataset.select([0])[0])
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
if __name__ == '__main__':
    main()