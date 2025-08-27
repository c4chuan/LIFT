import os.path
from PIL import Image
import cv2
import re
from typing import Any
from browser_env.utils import StateInfo, pil_to_b64, pil_to_vertex
from src.prompts.prompts import GUIDANCE,EXAMPLES,TEMPLATE
from browser_env.env_config import URL_MAPPINGS
from src.envs.actions import Action
from src.llms.utils import _add_modality_key_for_sglang_messages



class PromptConstructor:
    def __init__(self,save_dir):
        self.save_dir = save_dir

    def construct_messages(self,task, obs, info, guidance = None, examples = None):
        """
        需要的信息包括：
        1. examples：是否需要few-shot
        2. info：对任务的描述
        3. trajectory：历史的信息
        4. obs：对当前的环境的描述
        """
        messages = [] # 存储最终返回的input的messages
        messages = self._add_guidance(messages, guidance) # add guidance
        messages = self._add_examples(messages, examples) # add examples
        messages = self._add_query(messages,task,obs,info) # 组装query
        return messages


    def _add_guidance(self, messages, guidance):
        if guidance is not None:
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "text": GUIDANCE[guidance]
                        }
                    ]
                }
            )
        return messages

    def _add_examples(self, messages, examples):
        examples = EXAMPLES[examples]
        if examples is not None:
            for example in examples:
                messages += [
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": example['query'],
                            },
                            {
                                "image": example['image']
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "text": example['answer']
                            }
                        ]
                    }
                ]
        return messages

    def _add_query(self,messages,task,obs,task_info):
        url = task_info['page'].url
        image = obs['image']
        image_path = os.path.join(self.save_dir,str(task.task_id))
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        cv2.imwrite(image_path+f'/step_{str(task.steps)}_obs.png',cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        intent = task.task_info['intent']
        previous_action = self._get_pre_action(task.trajectory)
        query = TEMPLATE['LIFT'].format(url=url,intent=intent,previous_action=previous_action)
        messages.append({
            "role": "user",
            "content": [
                {
                    "text": query
                },
                {
                    "image": image_path+f'/step_{str(task.steps)}_obs.png'
                }
            ]
        })
        return messages


    def _get_pre_action(self,trajectory):
        if not len(trajectory):
            return 'None'
        else:
            prev_actions = ""
            for index,summary in enumerate(trajectory):
                # 去掉summary中的
                summary = summary.replace('<|vision_start|>','')
                summary = summary.replace('<|vision_end|>','')
                summary = summary.replace('<|image_pad|>','')
                summary = summary.replace('<image>', '')
                print(summary)
                prev_actions += f"summary of step {index}:{summary}\n"
            return prev_actions

    def extract_action(self,response):
        # find the first occurence of action
        pattern = rf"```((.|\n)*?)```"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        else:
            return None

    def exstract_summary(self,response):
        """
            提取所有被 <summary>...</summary> 包裹的内容，返回一个列表。
            """
        try:
            pattern = re.compile(r'<summary>(.*?)</summary>', re.DOTALL)
            return pattern.findall(response)[0]
        except:
            return ""