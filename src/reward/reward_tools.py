import re

def action_format_reward(response):
    # find the first occurence of action
    pattern = rf"```((.|\n)*?)```"
    match = re.search(pattern, response)
    if match:
        action_str = match.group(1).strip()
    else:
        return 0.0
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

def summary_format_reward(response):
    pattern = r'<summary>(.*?)</summary>'
    match = re.search(pattern, response)
    if match:
        return 1.0
    else:
        return 0.0

if __name__ == "__main__":
    r = summary_format_reward("""<summary>
Observation: The "Category" dropdown menu is open, and the "Xbox" category is visible among the options. The next step is to select the "Xbox" category to filter the listings.
</summary>""")
    print(r)