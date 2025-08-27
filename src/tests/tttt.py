import requests

messages = requests.get("http://localhost:7333/get_messages",params={"num":1})
with open("./resp.txt", "r", encoding="utf-8") as f:
    response = eval(f.read())
# response = ["Let's st``````afsdlkfajlkdwoiefqifn''afsdfa" for i in range(4)]
request_response = requests.post("http://localhost:7333/get_valid_action_rewards", json={"responses": response}).json()['rewards']

print(requests.post("http://localhost:7333/get_valid_action_rewards", json=response))
