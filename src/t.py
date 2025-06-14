import requests
from cupyx.scipy.ndimage import shift

from prompts.prompts import EXAMPLES
from prompts.prompt_construct import PromptConstructor
from examples.online_serving.openai_chat_completion_tool_calls_with_reasoning import messages
from reward.rewarder import Rewarder
def main():
    rewarder = Rewarder()
    shift_reward, zoom_reward = rewarder.reward(EXAMPLES['LIFT'][0]['answer'],"../data/example/example.png")
    print(shift_reward, zoom_reward)

if __name__ == '__main__':
    main()