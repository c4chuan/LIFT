from src.utils.data_tools import dataset_construct
from vwa.src.envs.browser import FastCachedwActionMatchingBrowserEnv
from concurrent.futures import ThreadPoolExecutor
from src.env.envtools import refresh_env_login,reset

class EnvironmentManager:
    def __init__(self, initial_configs):
        self.envs = [FastCachedwActionMatchingBrowserEnv(
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
        ) for _ in range(initial_configs.num_envs)]
        if initial_configs.initial_refresh_env:
            self.refresh_env()
    def refresh_env(self):
        """
        刷新环境
        """
        refresh_env_login()
        self.envs = self.parallel_areset(self.envs,dataset)
    def parallel_areset(self, envs,config_files):
        """
        多线程重置环境
        """
        with ThreadPoolExecutor(max_workers=len(envs)) as executor:
            futures = [executor.submit(env.reset()) for env in envs]
        return [future.result() for future in futures]


def get_initial_data(task_indices,dataset):
    """获取初始数据"""
    # 从dataset中获取task_id在task_indices中的数据，并组成可训练的数据返回
    return dataset[task_indices]
if __name__ == '__main__':
    dataset = dataset_construct()


