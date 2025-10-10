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
Based on the screenshot, here is the relevant part of the page with the buttons, elements, and lists:

1. **Keyword:**
   - [25] "Century Furniture English Roll Arm Sofa" - Laid down
   - [30] Complete Guitar Rig Full Size HSS Black
   - [31] Guitar Gig Bag with Keys
   - [40] Marshall APM 112 1x12 Compact Cabinet Amplifier

According to the observations above, the list does not contain a query option to filter for a specific price range or item attributes. It seems necessary to add a new method to filter based on color and price to satiate the goal. The hints are for a query box and other filter buttons. The aim is to find the best method to filter for attributes using the relevant applicable query or filters.

Proceeding with a new search by refreshing or using advanced search filters:
<action>
press [Enter]
</action>
"""
    print(format_reward_cal(r))

