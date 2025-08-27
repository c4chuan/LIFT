import requests

from prompts.prompts import EXAMPLES
import requests
# from src.reward_backend import rewarder
from multiprocessing import shared_memory
# import warnings
# warnings.filterwarnings('ignore', 'resource_tracker:UserWarning')


def main():
    # rewarder = DeepSpeedRewarder()
    # rewarder = Rewarder("/data/wangzhenchuan/models/Qwen2___5-VL-7B-Instruct")
    response = EXAMPLES['LIFT'][0]['answer']
    image_path = "../data/example/example.png"
    data = {"requests": [{"response": response, "image_path": image_path} for _ in range(5)]}
    result = requests.post("http://localhost:7452/rewards", json=data)
    print(result.text)

if __name__ == '__main__':
    main()