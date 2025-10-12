import re

def action_format_reward(response):
    # find the first occurence of action
    pattern = rf"<action>((.|\n)*?)</action>"
    match = re.search(pattern, response)
    if match:
        # 检查是否有多个匹配
        # 我们通过尝试使用 re.finditer 来判断，如果能找到第二个，就说明有多个
        # 从第一个匹配的结束位置开始搜索，看是否还有其他匹配
        remaining_response = response[match.end():]

        # 再次搜索，如果能找到另一个匹配，则说明有多个
        if re.search(pattern, remaining_response):
            return 0.0  # 找到多个匹配，返回0.0
        else:
            action_str = match.group(1).strip()
            # 这里你可以继续处理 action_str
            print(f"找到唯一匹配: \n{action_str}")
    else:
        return 0.0  # 没有找到匹配
    action_str = action_str.strip()
    if "[" in action_str:
        action = action_str.split("[")[0].strip()
    else:
        actions = action_str.split()
        if actions:
            action = actions[0].strip()
        else:
            return 0.0
    match action:
        case "click":
            match = re.search(r"click ?\[(\d+)\]", action_str)
            if not match:
                return 0.0
            element_id = match.group(1)
            return 1.0
        case "clear":
            match = re.search(r"clear ?\[(\d+)\]", action_str)
            if not match:
                return 0.0
            element_id = match.group(1)
            return 1.0
        case "hover":
            match = re.search(r"hover ?\[(\d+)\]", action_str)
            if not match:
                return 0.0
            return 1.0
        case "type":
            # add default enter flag
            if not (action_str.endswith("[0]") or action_str.endswith("[1]")):
                action_str += " [1]"

            match = re.search(
                r"type ?\[(\d+)\] ?\[(.+)\] ?\[(\d+)\]", action_str
            )
            if not match:
                return 0
            element_id, text, enter_flag = (
                match.group(1),
                match.group(2),
                match.group(3),
            )
            if enter_flag == "1":
                text += "\n"
            return 1.0
        case "press":
            match = re.search(r"press ?\[(.+)\]", action_str)
            if not match:
                return 0.0
            key_comb = match.group(1)
            return 1.0
        case "scroll":
            # up or down
            match = re.search(r"scroll ?\[?(up|down)\]?", action_str)
            if not match:
                return 0.0
            direction = match.group(1)
            return 1.0
        case "goto":
            match = re.search(r"goto ?\[(.+)\]", action_str)
            if not match:
                return 0.0
            url = match.group(1)
            return 1.0
        case "new_tab":
            return 1.0
        case "go_back":
            return 1.0
        case "go_forward":
            return 1.0
        case "tab_focus":
            match = re.search(r"tab_focus ?\[(\d+)\]", action_str)
            if not match:
                return 0.0
            page_number = int(match.group(1))
            return 1.0
        case "close_tab":
            return 1.0
        case "stop":  # stop answer
            match = re.search(r"stop ?\[(.+)\]", action_str)
            if not match:  # some tasks don't require an answer
                answer = ""
            else:
                answer = match.group(1)
            return 1.0
    return 0.0

def get_action_id_answer(response):
    # find the first occurence of action
    pattern = rf"<action>((.|\n)*?)</action>"
    match = re.search(pattern, response)
    action_info = {
        "element_id": -1,
        "answer": None,
        "url": None,
    }
    if match:
        action_str = match.group(1).strip()
    else:
        return action_info
    action_str = action_str.strip()
    if "[" in action_str:
        action = action_str.split("[")[0].strip()
    else:
        actions = action_str.split()
        if actions:
            action = actions[0].strip()
        else:
            return action_info
    match action:
        case "click":
            match = re.search(r"click ?\[(\d+)\]", action_str)
            if not match:
                return action_info
            element_id = match.group(1)
            action_info["element_id"] = element_id
            return action_info
        case "clear":
            match = re.search(r"clear ?\[(\d+)\]", action_str)
            if not match:
                return 0.0
            element_id = match.group(1)
            action_info["element_id"] = element_id
            return action_info
        case "hover":
            match = re.search(r"hover ?\[(\d+)\]", action_str)
            if not match:
                return action_info
            return action_info
        case "type":
            # add default enter flag
            if not (action_str.endswith("[0]") or action_str.endswith("[1]")):
                action_str += " [1]"

            match = re.search(
                r"type ?\[(\d+)\] ?\[(.+)\] ?\[(\d+)\]", action_str
            )
            if not match:
                return action_info
            element_id, text, enter_flag = (
                match.group(1),
                match.group(2),
                match.group(3),
            )
            if enter_flag == "1":
                text += "\n"
            action_info["element_id"] = element_id
            return action_info
        case "press":
            match = re.search(r"press ?\[(.+)\]", action_str)
            if not match:
                return action_info
            key_comb = match.group(1)
            return action_info
        case "scroll":
            # up or down
            match = re.search(r"scroll ?\[?(up|down)\]?", action_str)
            if not match:
                return action_info
            direction = match.group(1)
            return action_info
        case "goto":
            match = re.search(r"goto ?\[(.+)\]", action_str)
            if not match:
                return action_info
            url = match.group(1)
            action_info["url"] = url
            return action_info
        case "new_tab":
            return action_info
        case "go_back":
            return action_info
        case "go_forward":
            return action_info
        case "tab_focus":
            match = re.search(r"tab_focus ?\[(\d+)\]", action_str)
            if not match:
                return action_info
            page_number = int(match.group(1))
            return action_info
        case "close_tab":
            return action_info
        case "stop":  # stop answer
            match = re.search(r"stop ?\[(.+)\]", action_str)
            if not match:  # some tasks don't require an answer
                answer = ""
            else:
                answer = match.group(1)
                action_info["answer"] = answer
            return action_info
    return action_info

def format_reward_cal(response):
    format_reward = 0.0
    pattern = r'<summary>([\s\S]*?)</summary>'
    match = re.search(pattern, response)
    if match:
        format_reward += 1.0
    pattern = r'<zoom in>([\s\S]*?)</zoom in>'
    match = re.search(pattern, response)
    if match:
        format_reward += 1.0
    pattern = r'<shift>([\s\S]*?)</shift>'
    match = re.search(pattern, response)
    if match:
        format_reward += 1.0
    return format_reward + action_format_reward(response)

if __name__ == "__main__":
    r = """
Let's observe step-by-step. First, I will zoom in to observe the whole page.
<zoom in>
This page can be divided into the following sections:

1. Navigation Section (top):
   - Includes links for "My account" and "Logout."
   - References specific query headers such as "sCategory."

2. Search Input Section:
   - Includes input boxes for searching (Your search), and defining the location (City).

3. Order Selection Section:
   - Displays the option to order items by "Higher price first."

4. Listings Display Section:
   - Lists several watch items including specific details such as:
     - Model, price, location, and condition.
     - Includes relevant product information, e.g., antiques (scrollable).

5. Product Images & Shipping Options:
   - Some products (adjusted with a marker) appear in images despite their detailed description.

</zoom in>
According to the observation above, this page displays information about watches but does not include any image freely aligned with its listings. I need to focus on the actual watch listings to figure out the information about the last name of the lister who posted an item that does not include jewelry in its image. Then, I will shift to examine the image lists for its alignment with the listings.
<shift>
We move our focus to the Listings Display Section on the right side of the webpage:
The Listings Display is displayed on a white background with several products or items are placed beside each listing heading; Let's see if any of the Previous watch  items symbolized by an adjusted marker in Bounding box No. 21, 25, 26, and 27 is an example of an item that does not include jewelry in its image, amidst the listings of watch items including a specific model and description like "all category."

According to the observation above, the products displayed are all watch items, not jewelry items. I will need twoto check the last name of the lister who posted the watch item as per this observation on the displayed products.
</shift>
The Products displayed are all watch items and the
<zoom in>
Zooming in on each marker to analyze the image further, the products appear to have hashtags and references to Rolex or other watch brands.
</zoom in>
According to the observation above, all products are clearly watch-related with European or American brands such as Rolex. Since the goal is to find jewelry in the last name of the person who listed without an image, the selection results do not match the requirement as they all appear to be unrelated.
<shift>
These products, like Rolex dated northwest of 2010, are all contained with default captured images matching respective brands. Shift to navigate next or switch the 'category' type with tagged filters such as "not jewelry but watch".
</shift>
By neglecting the observation, I will find the relevant listing in the 'Category' fouIndicates needing further operations to refocus on the search results that do not match the goal. I need to implied correctly and look for the right criteria with an appropriate equipment like 'Category' or 'not jewelry but watch'.
<zoom in>
Boundaries mirror all belonging to previous wax environments insights to correctly retiring equate focus on relevant listings matching jewelry search conclusive logic.
</zoom in>
Previous actions and current view involve mismatched listings.
<shift>
Observation done correctly involving edit recent steps and responsive to guide for unordered list and a focus on inclusion accuracy:
</shift>
The product descriptions and labels, under the "Listings Display Area," still all continue to be misaligned with the goal. The environment needs a closer look at the adjusting image relative to listings.
<zoom in>
By zooming in on the current listings, you can identify the specific watch that should be examined to align with the 'not jewelry but watch' label.
</zoom in>
It seems the products under the current 'Listings Display' area align well with cameras or watches and do not match the goal of matching the last name of the lister who posted a jewelry item.
<shift>
Observing the last name of the lister in the right environment makes the product observation confused to navigate listings per 'not jewelry but watch.'
</shift>
Observation shows there are no jewelry tags or香港是 searching in the criterion, the product matches must be returned right to ticker and the observation will correct.
<zoom in>
By exploring the listings, we see all product images of 'Rolex' or luxury watch types, not jewelry. The task completion needs logging the correct condition, aim, or category.
</zoom in>
To correctly capture the last name of listers for what states should not be an issue with focus on the precision in interpreting listed products.
<shift>
In reflection, necessità means the product is must have alphabetical character sequence or number derived from list listing title.
</shift>
The last name of the lister, which does not include jewelry in its image, does not meet the conditions stated or included on the current page. Observations indicate all listed items are watches, not jewelry.
<summary>
Observations so far:
1. The page displays various watch items, not jewelry.
2. Product images related to watches are listed.

Following correct analysis, the products are all categorized as watches and do not match the requirement of jewelry.

The next action should be addressing the actual presence of wristwatches with a distinct absence of mandate to preclude matches with jewelry listings.

The next action will be to explore specific product attributes and defining removal condition.
</summary>

<action>
type [id] [text] [0]
</action>
"""
    print(format_reward_cal(r))

