import requests
from difflib import SequenceMatcher
BASE_URL = "http://localhost:7333"

responses = ["<summary>'Boats'</summary>```click [38]```"]
resp = requests.get(f"{BASE_URL}/get_messages", params={"num": 1})
requests.post(url=f"{BASE_URL}/get_valid_action_rewards", json={"responses":responses}).json()
requests.post(url=f"{BASE_URL}/feed_responses", json=responses).json()
