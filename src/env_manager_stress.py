import time
import requests

BASE_URL = "http://localhost:7333"

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
    num = 2
    # 1. 先从服务里拉取消息
    messages = test_get_messages(num)

    # 2. 构造对应的“模型回复”填充测试
    #    这里假设环境 manager 构造的 messages 长这样，你可以根据实际格式调整
    #    例如，如果 LLM 期望的 action 字符串为 "click_button_5"，则填入相应值
    dummy_responses = [
        "```click [5]```",
        "```click [10]```",
        "```click [3]```",
        "```scroll down```",
        "```stop [aaa]```"
    ]
    counter = 0
    for _ in range(100):
        # 3. 发送到 feed_responses（异步后台触发）
        test_feed_responses([dummy_responses[counter%len(dummy_responses)] for _ in range(num)])
        time.sleep(20)

        # 4. 再次拉取下一批消息
        next_msgs = test_get_messages(num)
        time.sleep(5)
        print("Next batch:", next_msgs)
        counter+=1
        if counter % 20 == 0:
            requests.get(f"{BASE_URL}/refresh_env")
