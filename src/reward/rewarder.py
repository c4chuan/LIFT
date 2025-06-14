import os.path

import torch,re,cv2
import torch.nn.functional as F
import numpy as np
from ray.experimental.array.remote import zeros_like

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor,AutoTokenizer,AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from src.utils.visualize_tools import show_mask_on_image


class Rewarder:
    def __init__(self, model_path = "/data/wangzhenchuan/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-7B-Instruct"):
        tokenizer, model, processor, context_len = self._load_vlm_model(model_path)
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor

    def _load_vlm_model(self, model_path):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            attn_implementation="eager",
            device_map="auto",
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if hasattr(model.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048

        return tokenizer, model, processor, context_len

    def batch_reward(self,responses, image_paths, visualize = False, visual_save = None):
        """基于reward函数的相同逻辑，并行处理多个response"""


    def reward(self,response,image_path,visualize = False,visual_save = None):
        """
        response: 模型输出的text tokens
        image: 对应的环境的截图
        """
        # reset peak-memory stats on this device
        device = self.model.device
        if torch.cuda.is_available() and device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        # 1. 从response中提取标签,组装新的input_text,并提取出观察序列
        processed_input,obs_seq = self._get_processed_input(response)

        # 2. 组装messages
        messages = self._make_messages(processed_input,image_path)

        # 3. 输出attention
        outputs = self._foward_once(messages)
        outputs['aggregate_attn'] = self._aggregate_attentions(outputs)

        # 4. 计算reward
        shift_reward,zoom_reward = self._compute_reward(obs_seq,processed_input,outputs,visualize,visual_save)
        # query peak GPU memory
        if torch.cuda.is_available() and device.type == 'cuda':
            peak_bytes = torch.cuda.max_memory_allocated(device)
            peak_mib = peak_bytes / (1024 ** 2)
            print(f"[Rewarder] Peak GPU memory during reward(): {peak_mib:.1f} MiB")

        return shift_reward,zoom_reward

    def _get_processed_input(self,response):
        """提出标签内的内容"""
        # 用一个正则同时匹配两种标签，并捕获标签名和内容
        pattern = re.compile(r"<(?P<tag>zoom in|shift)>(?P<content>.*?)</(?P=tag)>", re.DOTALL)

        # finditer 返回 Match 对象的迭代器
        results = [(m.group("tag"), m.group("content")) for m in pattern.finditer(response)]

        # 组装input
        processed_input = ''.join([f"{content}" for tag, content in results])

        return processed_input,results

    def _make_messages(self,text,image_path):
        """组装messages"""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]
        return messages

    def _foward_once(self,messages):
        """一次前向传播"""
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        processed_images = inputs["processed_images"]
        # 再让inputs删掉processed_images
        inputs.pop("processed_images")
        num_patches = int(torch.prod(inputs["image_grid_thw"]) / (2 ** 2))
        image_grid = inputs["image_grid_thw"]
        inputs = inputs.to(self.model.device)
        input_ids = inputs["input_ids"]
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                use_cache=False,
                return_dict_in_generate=True,
                output_attentions=True
            )

        results = {
            "attention": outputs['attentions'],
            "num_patches": num_patches,
            "image_grid": image_grid,
            "processed_image": processed_images[0],
            "text": text,
            "input_ids": input_ids
        }
        return results

    def _aggregate_attentions(self,outputs):
        """attn[0][0]是一个长度为层数的列表，每个元素是size为[1,28,N,N]的tensor
        将每层attention先按注意力头平均再按层平均，最后得到NxN的矩阵"""
        attn = outputs['attention']
        # 将每层attention堆叠成[L, heads, N, N]
        attn_tensor = torch.zeros_like(attn[0][0].squeeze(0))
        for att in attn[0]:
            attn_tensor+= att.squeeze(0)
        # 对head维度求平均 -> [L, N, N]
        head_avg = attn_tensor/len(attn[0])
        # 对layer维度求平均 -> [N, N]
        layer_avg = head_avg.mean(dim=0)
        return layer_avg

    def _get_obs_indices(self,full_text,obs_seq):
        """obs_token_indices[i] 即是第 i 段 obs（tag=shift/zoom in）在完整文本对应的 token 下标序列"""
        # —— 先对完整的 full_text 做一次分词，获取 offset_mapping ——
        encoding_full = self.tokenizer(
            full_text,
            return_offsets_mapping=True,
            add_special_tokens=False  # 保证 offset_mapping 与 full_input_ids 一一对应
        )
        full_input_ids = encoding_full["input_ids"]  # 整个文本的 token IDs（长度 N）
        offsets = encoding_full["offset_mapping"]  # 长度也是 N，每个元素是 (char_start, char_end)

        # —— 逐段定位 obs 在 full_text 里的字符区间，并据此找到它们对应的全局 token 下标 ——
        obs_token_indices = []  # will be a list of lists，obs_token_indices[i] = [token_idx_1, token_idx_2, ...]
        for tag, content in obs_seq:
            # 在 full_text 中查找 content 的起止字符索引
            start_char = full_text.find(content)
            if start_char < 0:
                raise ValueError(f"在完整文本中找不到 obs 内容：{content}")
            end_char = start_char + len(content)

            # 遍历 offsets，将所有 char_start >= start_char 且 char_end <= end_char 的 token idx 加入
            matched_indices = []
            for idx, (char_s, char_e) in enumerate(offsets):
                if char_s >= start_char and char_e <= end_char:
                    matched_indices.append(idx)
            if len(matched_indices) == 0:
                raise ValueError(f"无法找到任何 token 完全落在子串范围内：{content}")
            st = matched_indices[0]
            ed = matched_indices[-1]+1
            obs_token_indices.append((st,ed,full_input_ids[st:ed]))
        return obs_token_indices,full_input_ids

    def _get_attn_seq(self,obs_attn,obs_range_seq,outputs):
        """获取每段观察文本对image tokens的attention"""
        # 首先获取每段obs的范围，比如第一段obs就应该是从0到len(obs[0])
        obs_attn_seq = []
        for index,(st,ed,slice_ids) in enumerate(obs_range_seq):
            # 提取当前段obs对应的attention切片，shape为[length, N_i]

            slice_attn = obs_attn[st:ed]

            # 可视化debug(可删除)
            # for i in range(slice_attn.shape[0]):
            #     token_id = slice_ids[i]
            #     token_attn = slice_attn[i]
            #     token_attn_image = self.visualize(outputs,token_attn.cpu())
            #     token = self.tokenizer.decode(token_id)
            #     save_dir = f'../results/token_level_debug/obs_{index}'
            #     if not os.path.exists(save_dir):
            #         os.mkdir(save_dir)
            #     save_path = f'../results/token_level_debug/obs_{index}/{i}_{token}.png'
            #     cv2.imwrite(save_path,token_attn_image)

            # 其次，每段obs的每个token对图片的attention全部加起来得到size为[N_i]的tensor
            # 求平均即得到每段obs对所有image tokens的attention
            mean_attn = slice_attn.mean(dim=0)

            obs_attn_seq.append(mean_attn.cpu())
        return obs_attn_seq

    def _compute_reward(self,obs_seq,processed_input,outputs,visualize,visual_save):
        """根据观察序列和这些text tokens关于image tokens的attention计算reward"""

        attn = outputs['aggregate_attn']
        num_patches = outputs['num_patches']
        processed_image = outputs['processed_image']
        # 首先将obs 用batch的形式得到token ids
        obs_tag_seq = [tag for tag,_ in obs_seq]
        obs_range_seq,full_input_ids = self._get_obs_indices(processed_input,obs_seq)

        # 输入的整个input的组织形式应该如下所示：
        # <|im_start|>system
        # You are a helpful assistant.<|im_end|>
        # <|im_start|>user
        # <|vision_start|><|image_pad|><|vision_end|>
        # ......
        # <|im_end|>
        # <|im_start|>assistant

        vision_start = len(self.tokenizer(outputs['text'].split("<|image_pad|>")[0], return_tensors='pt')["input_ids"][0])
        vision_end = vision_start+num_patches
        obs_attn = attn[vision_end+1:vision_end+1+len(full_input_ids), vision_start:vision_end] # obs_attn的形状应该是 obs文本的长度*图片tokens的长度
        obs_attn_seq = self._get_attn_seq(obs_attn,obs_range_seq,outputs)

        # 初始化history为均匀分布
        N = obs_attn_seq[0].size(0)
        history = torch.zeros_like(obs_attn_seq[0])

        shift_rewards = 0 # 初始化shift_reward
        zoom_rewards = 0 # 初始化zoom_reward

        for index,tag in enumerate(obs_tag_seq):

            cur_attn = obs_attn_seq[index]
            # 首先，obs_seq每一个元素都是一段观察的文本+对应的tag(<zoom in>还是<shift>)
            if tag == 'shift':
                # 计算该段obs与history的KL散度
                if index == 0:
                    avg_history = torch.ones_like(obs_attn_seq[0])/N
                else:
                    avg_history = history/index
                comp_avg_history = avg_history.max() - avg_history
                shift_rewards += self._containing_degree(cur_attn,comp_avg_history)

            elif tag == 'zoom in':
                # 计算该段obs的attention与上一段obs的attention的包含度，即余弦相似度
                if index == 0:
                    avg_history = torch.ones_like(obs_attn_seq[0])/N
                    zoom_reward = self._containing_degree(cur_attn,avg_history)
                else:
                    prev_attn = obs_attn_seq[index-1]
                    zoom_reward = self._containing_degree(cur_attn,prev_attn)

                zoom_rewards += zoom_reward

            # 计算后，将该段obs的attention加入历史
            history += obs_attn_seq[index]

            # 是否需要可视化
            if visualize:
                heated_image = self.visualize(outputs,obs_attn_seq[index])
                if visual_save:
                    if not os.path.exists(visual_save):
                        os.mkdir(visual_save)
                    cv2.imwrite(f'{visual_save}/{tag}_{index}.png',heated_image)


        # log平缓
        shift_rewards,zoom_rewards = self._log_smooth(shift_rewards,zoom_rewards)
        return shift_rewards, zoom_rewards

    def _log_smooth(self,shift_rewards,zoom_rewards):
        """log平缓"""
        return np.log2(shift_rewards), np.log2(zoom_rewards)

    def _compute_shift_reward(self,obs_attn,avg_history_attn):
        """计算与历史的补集的包含度"""
        return self._containing_degree(obs_attn,avg_history_attn)


    def visualize(self,outputs,attn):
        """可视化这一整段obs对于图片的Attention"""
        image_grid = outputs['image_grid']
        processed_image = outputs['processed_image']

        attn = attn/ attn.sum()
        attn = attn.reshape(int(image_grid[0][1]/2), int(image_grid[0][2]/2))
        attn_over_image = np.kron(attn, np.ones((28,28)))
        attn_over_image = attn_over_image/attn_over_image.max()
        # 从image_path读取图片并转为np.array
        processed_image = np.uint8(np.array(processed_image)*255)
        img_with_attn, heatmap = show_mask_on_image(processed_image, attn_over_image)
        # 转换一下颜色
        img_with_attn = cv2.cvtColor(img_with_attn, cv2.COLOR_RGB2BGR)

        return img_with_attn

    def _containing_degree(self,x, y):
        """
        计算两个集合之间的包含度

        containing_degree= sum_i min(x_i, y_i) / sum(x)

        要求：
          - x.shape == y.shape
          - 所有元素都 >= 0

        参数:
          - x: torch.Tensor，1D，长度为 N，元素非负
          - y: torch.Tensor，1D，长度为 N，元素非负

        返回:
          - torch.Tensor，标量，加权 Jaccard 相似度
        """
        # 检查形状一致
        if x.shape != y.shape:
            raise ValueError(f"输入张量形状不一致：x.shape={x.shape}, y.shape={y.shape}")
        # 确保都是 1D（如果更高维，可以先 flatten）
        if x.dim() != 1:
            x = x.view(-1)
            y = y.view(-1)

        # 检查非负性
        if (x < 0).any() or (y < 0).any():
            raise ValueError("加权 Jaccard 要求输入 tensor 中的元素均为非负数。")

        # 计算对应元素的 min 和 max，并累加
        intersect = torch.min(x, y).sum()
        union = x.sum()

        return intersect / union
