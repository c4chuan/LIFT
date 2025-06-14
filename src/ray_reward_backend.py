import ray
import torch
from reward.rewarder import Rewarder

@ray.remote(num_gpus=1)
class RewarderActor:
    def __init__(self, model_path=None):
        # 该构造函数在各自的 GPU 进程里执行一次，加载模型到对应 GPU
        self.rewarder = Rewarder(model_path or "/path/to/model")

    def reward(self, response: str, image_path: str, visualize=False, visual_save=None):
        # 将调用转到 Rewarder.reward，并返回结果
        return self.rewarder.reward(response, image_path, visualize, visual_save)
if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    ray.init(num_gpus=num_gpus)
    # 按 GPU 数量创建同样数量的 RewarderActor，每个占用一块 GPU
    actors = [
        RewarderActor.options(num_gpus=1).remote(model_path="/path/to/model")
        for _ in range(num_gpus)
    ]


