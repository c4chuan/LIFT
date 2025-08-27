import requests
def test_get_rewards(responses,images):
    data = {"requests": [
        {"response": responses[i], "image_path": images[i]} for i in
        range(len(responses))]}

    results = requests.post(f"http://localhost:7452/rewards", json=data)

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
```
click [14]
```
</summary>"""
dummy_responses = [
    t_response
    for i in range(2)
]

images = ["/data/wangzhenchuan/Projects/LIFT/src/0/step_0_obs.png" for j in range(2)]

test_get_rewards(dummy_responses, images)