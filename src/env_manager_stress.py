import time
import random

import requests

BASE_URL = "http://localhost:7333"

def test_refresh_env():
    """
    POST 到 /refresh_env，触发后台处理，不等待结果。
    """
    resp = requests.get(
        f"{BASE_URL}/refresh_env"
    )
    resp.raise_for_status()
    print("✅ refresh_env accepted:", resp.json())

def test_get_messages(num=2, timeout=20):
    """
    循环调用 /get_messages 接口，直到拿到至少 num 条消息或超时。
    返回消息列表。
    """
    start = time.time()
    while True:
        resp = requests.get(f"{BASE_URL}/get_messages", params={"num": num})
        resp.raise_for_status()
        data = resp.json()
        msgs = data.get("messages", [])
        if len(msgs) >= num:
            print(f"✅ Received {len(msgs)} messages:")
            for i, m in enumerate(msgs, 1):
                print(f"  {i}. {m}")
            return msgs
        if time.time() - start > timeout:
            raise TimeoutError(f"Timeout after {timeout}s, only got {len(msgs)} messages")
        print("⏳ Not enough messages yet, retrying...")
        time.sleep(0.5)
def test_get_rewards(responses,images):
    data = {"requests": [
        {"response": responses[i], "image_path": images[i]} for i in
        range(len(responses))]}

    results = requests.post(f"http://localhost:7452/rewards", json=data)

    return results


def test_get_valid_action_rewards(responses):
    data = {"responses": responses}
    results = requests.post(f"http://localhost:7333/get_valid_action_rewards", json=data)
    return results

def test_feed_responses(responses):
    """
    POST 到 /feed_responses，触发后台处理，不等待结果。
    """
    resp = requests.post(
        f"{BASE_URL}/feed_responses",
        json=responses
    )
    resp.raise_for_status()
    print("✅ feed_responses accepted:", resp.json())

if __name__ == "__main__":
    num = 1
    # 1. 先从服务里拉取消息
    messages = test_get_messages(num)

    # 2. 构造对应的“模型回复”填充测试
    #    这里假设环境 manager 构造的 messages 长这样，你可以根据实际格式调整
    #    例如，如果 LLM 期望的 action 字符串为 "click_button_5"，则填入相应值

    t_response = f"""Let's observe step by step. First, I will zoom in to see more of the video gamer commitments.
<zoom in>
This screenshot has four columns. It is in the Us classify section. The listings are different

### Column A:
- **Label:** 'Your search' and 'Please search for'
- Boxes: It shows 7 items.

### Column B:
- **Label:** 'City'
- Boxes: shows checkboxes and dropdowns

### Column C:
- **Label:** 'Show only listings with pictures'
- Boxes: it shows a box with a picture and text boxes for pictures and price

### Column D:
- **Label:** 'Price Min.'
  - Text: includes different prices
- Box: it includes price range checkboxes

</zoom in>
According to the observation above, the listings shown here are related to Video Gaming, and not Motorcycles. Next, I need to shift focus to the elements that can navigate or adjust the clasification which I suspect can allow switching into Matching the right classification.

<shift>{"pad|"*8000}</shift>
<summary>
Observations so far:
- The screenshot is in the 'Video gaming' section.
- The goal is to find motorcycle listings according to the objective.
- The initiation action was clicking 'Video gaming', so we need a next step to adjust the clasification. Since there are steering options, I'll check elements that might navigate to another categorization.

Action: Click the 'Video gaming' option in column C to change from the current topic.
</summary>
<action>
stop [http://localhost:9980/index.php?page=item&id=40404 and http://localhost:9980/index.php?page=item&id=38125]
</action>
"""


    dummy_responses = [
        t_response,
    ]
    counter = 0
    for ti in range(100):
        # 3. 发送到 feed_responses（异步后台触发）
        response_model = {
            "response": t_response,
            "reward_sum": random.uniform(0, 3)
        }
        reward_responses = [response_model for _ in range(num)]
        images = ["/data/wangzhenchuan/Projects/LIFT/src/0/step_0_obs.png" for i in range(num)]
        test_get_valid_action_rewards([{"response": t_response,"task_id": 7} for _ in range(4)])
        test_feed_responses(reward_responses)
        time.sleep(5)
        # if ti%3 == 0:
        #     print("Refreshing env...")
        #     test_refresh_env()
        # 4. 再次拉取下一批消息
        next_msgs = test_get_messages(num)
        time.sleep(5)
        # print("Next batch:", next_msgs)
        counter+=1
        # if counter % 20 == 0:
        #     requests.get(f"{BASE_URL}/refresh_env")
