import os.path

import cv2

from src.prompts.prompts import GUIDANCE,EXAMPLES,TEMPLATE

class PromptConstructor:
    def __init__(self,save_dir):
        self.save_dir = save_dir

    def construct_prompt(self,task, obs, info, trajectory, guidance = None, examples = None):
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
        messages = self._add_trajectory(messages,trajectory) # add trajectory
        messages = self._add_query(messages,task,obs,info,trajectory) # 组装query
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

    def _add_query(self,messages,task,obs,task_info,trajectory):
        url = task_info['page'].url
        image = obs['image']
        image_path = os.path.join(self.save_dir,str(task['task_id']))
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        cv2.imwrite(image_path+f'/{str(len(trajectory))}.png',cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        intent = task['intent']
        previous_action = self._get_pre_action(trajectory)
        query = TEMPLATE['LIFT'].format(url=url,intent=intent,previous_action=previous_action)
        messages.append({
            "role": "user",
            "content": [
                {
                    "text": query
                },
                {
                    "image": image_path+f'/{str(len(trajectory))}.png'
                }
            ]
        })
        return messages

    def _add_trajectory(self,messages,trajectory):

        for his in trajectory:
            # TODO：增加trajectory逻辑
            messages.append()

        return messages

    def _get_pre_action(self,trajectory):
        if not len(trajectory):
            return 'None'
        else:
            return trajectory[-1]['action']