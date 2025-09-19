import os.path

import torch,re,cv2
import torch.nn.functional as F
import numpy as np
import deepspeed
from src.prompts.prompts import EXAMPLES
from ray.experimental.array.remote import zeros_like
import wandb

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor,AutoTokenizer,AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from src.utils.visualize_tools import show_mask_on_image, visualize_tensor_distribution, plot_1d_tensor
from src.reward.reward_tools import action_format_reward,summary_format_reward

# 可选的 accelerate 导入
try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

BASE_URL = "http://192.168.1.6:7333"

class Rewarder:
    def __init__(self, model_path = "/data/wangzhenchuan/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-7B-Instruct", use_accelerate=False):
        self.use_accelerate = use_accelerate
        self.accelerator = None
        
        if use_accelerate:
            if not ACCELERATE_AVAILABLE:
                raise ImportError("accelerate 库未安装，无法使用多卡模式。请运行: pip install accelerate")
            self.accelerator = Accelerator()
        
        tokenizer, model, processor, context_len = self._load_vlm_model(model_path)
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor

    def _load_vlm_model(self, model_path):
        if self.use_accelerate:
            # 使用 accelerate 时，使用 fp16 和 device_map="auto"
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                attn_implementation="eager",
                device_map="auto",
                trust_remote_code=True
            )
            # 使用 accelerate 准备模型
            model = self.accelerator.prepare(model)
        else:
            # 原始单卡模式
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
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


    def _compute_format_reward(self,response):
        """计算格式奖励"""
        return action_format_reward(response),summary_format_reward(response)

    def reward(self,response,image_path,visualize = False,visual_save = None):
        """
        response: 模型输出的text tokens
        image: 对应的环境的截图
        """
        # 如果使用多进程且不是主进程，则参与计算但不返回结果
        if self.use_accelerate and not self.accelerator.is_main_process:
            # 非主进程仍然参与前向传播计算，但不进行最终的 reward 计算和可视化
            self._participate_in_computation(response, image_path)
            return 0, 0, 0, 0
        
        # reset peak-memory stats on this device
        if self.use_accelerate:
            device = self.accelerator.device
        else:
            device = self.model.device
            
        if torch.cuda.is_available() and device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        a_format_reward,s_format_reward = self._compute_format_reward(response)

        # 1. 从response中提取标签,组装新的input_text,并提取出观察序列
        processed_input,obs_seq = self._get_processed_input(response)

        if len(obs_seq)== 0:
            # 如果没有观察序列，可以直接返回了
            return 0,0,a_format_reward,s_format_reward

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

        return shift_reward,zoom_reward,a_format_reward,s_format_reward

    def _participate_in_computation(self, response, image_path):
        """非主进程参与计算但不返回结果"""
        processed_input, obs_seq = self._get_processed_input(response)
        if len(obs_seq) == 0:
            return
        messages = self._make_messages(processed_input, image_path)
        # 只进行前向传播，让所有进程参与分布式计算
        self._foward_once(messages)

    def _get_processed_input(self,response):
        """提出标签内的内容"""
        # 用一个正则同时匹配两种标签，并捕获标签名和内容
        pattern = re.compile(r"<(?P<tag>zoom in|shift)>(?P<content>.*?)</(?P=tag)>", re.DOTALL)

        # finditer 返回 Match 对象的迭代器
        results = []
        for m in pattern.finditer(response):
            if len(m.group("content"))>5:
                results.append((m.group("tag"), m.group("content")))

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
        
        # 根据是否使用 accelerate 决定设备处理方式
        if self.use_accelerate:
            inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}
        else:
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

        # 如果使用 accelerate，可能需要收集分布式的 attention
        attentions = outputs['attentions']
        if self.use_accelerate and self.accelerator.num_processes > 1:
            attentions = self._gather_attentions(attentions)

        results = {
            "attention": attentions,
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

    def _gather_attentions(self, attentions):
        """使用 Accelerate 收集所有 GPU 的 attention"""
        if not self.use_accelerate or self.accelerator.num_processes <= 1:
            return attentions
            
        gathered_attentions = []
        for layer_attn in attentions[0]:  # attentions[0] 是第一个生成步骤的 attention
            # 收集所有进程的 attention
            gathered = self.accelerator.gather(layer_attn)
            gathered_attentions.append(gathered)
        return [tuple(gathered_attentions)]

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
            else:
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
                shift_reward = self._containing_degree(cur_attn,comp_avg_history)
                shift_rewards += shift_reward

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
                heated_image = self.visualize(outputs, obs_attn_seq[index])
                if visual_save:
                    if not os.path.exists(visual_save):
                        os.mkdir(visual_save)
                    if tag == 'zoom in':
                        r = zoom_reward
                        cv2.imwrite(f'{visual_save}/{tag}_{index}_{r}.png', heated_image)
                    if tag == 'shift':
                        r = shift_reward
                        cv2.imwrite(f'{visual_save}/{tag}_{index}_{r}.png', heated_image)


        # log平缓
        shift_rewards,zoom_rewards = self._log_smooth(shift_rewards,zoom_rewards)
        return shift_rewards, zoom_rewards

    def _log_smooth(self,shift_rewards,zoom_rewards):
        """log平缓,加一是为了把值域放到0~正无穷"""
        return np.log2(shift_rewards+1.0), np.log2(zoom_rewards+1.0)

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

class ChunkRewarder(Rewarder):
    def __init__(self, chunk_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_size = chunk_size

    def uniform(self,x):
        """将tensor转化成z-score分布"""

        def stretch_middle(x, gamma=2.0):
            x_min, x_max = x.min(), x.max()
            x_norm = (x - x_min) / (x_max - x_min)
            y = x_norm ** gamma
            return y * (x_max - x_min) + x_min

        def softmax_tensor(x):
            # 减去最大值防止数值溢出
            x_exp = torch.exp(x - torch.max(x))
            return x_exp / torch.sum(x_exp)

        def min_max_scale(x):
            return (x - x.min()) / (x.max() - x.min())

        def normalize(x):
            return x / x.sum()

        return normalize(min_max_scale(x))

    def instruction_func(self,score):
        if score < 0.5:
            return score
        else:
            return 1-score

    def _ratio_score(self,x,y):
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

    def _containing_score(self,x, y):
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
        comp_x = x.max()-x
        comp_x = comp_x / comp_x.sum()
        intersect = torch.min(comp_x, y).sum()
        union = x.sum()

        return 1 - (intersect / union)

    def _reward_func(self,x,y):
        ratio_score = self._ratio_score(x, y)
        inst_ratio_score = 2 * self.instruction_func(ratio_score)  # 乘以2是为了将值域缩放到[0,1]
        contain_score = self._containing_score(x, y)
        reward = inst_ratio_score * contain_score
        return reward, ratio_score, inst_ratio_score, contain_score

    def _log_smooth(self,shift_rewards,zoom_rewards):
        """log(动作的次数)*平均奖励"""
        shift_times_log = np.log(len(shift_rewards)+1)
        zoom_times_log = np.log(len(zoom_rewards)+1)
        shift_sum_ex = [p for p in shift_rewards if p != -1]
        zoom_sum_ex = [p for p in zoom_rewards if p != -1]
        if len(shift_sum_ex) == 0:
            shift_mean = 0
        else:
            shift_mean = np.mean(shift_sum_ex)
        if len(zoom_sum_ex) == 0:
            zoom_mean = 0
        else:
            zoom_mean = np.mean(zoom_sum_ex)
        shift_reward = shift_times_log * shift_mean
        zoom_reward = zoom_times_log * zoom_mean
        return shift_reward,zoom_reward

    def reward(self,response,image_path,visualize = False,visual_save = None):
        """
        response: 模型输出的text tokens
        image: 对应的环境的截图
        """
        # reset peak-memory stats on this device
        device = self.model.device
        if torch.cuda.is_available() and device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        a_format_reward,s_format_reward = self._compute_format_reward(response)

        # 1. 从response中提取标签,组装新的input_text,并提取出观察序列
        processed_input,obs_seq = self._get_processed_input(response)

        if len(obs_seq)== 0:
            # 如果没有观察序列，可以直接返回了
            return 0,0,a_format_reward,s_format_reward

        # 4. 计算reward
        shift_reward,zoom_reward = self._compute_reward(obs_seq,processed_input,image_path,visualize,visual_save)
        # query peak GPU memory
        if torch.cuda.is_available() and device.type == 'cuda':
            peak_bytes = torch.cuda.max_memory_allocated(device)
            peak_mib = peak_bytes / (1024 ** 2)
            print(f"[Rewarder] Peak GPU memory during reward(): {peak_mib:.1f} MiB")

        return 1.5*shift_reward,1.5*zoom_reward,0.1*a_format_reward,0.1*s_format_reward

    def _aggregate_attentions(self,attn):
        """attn[0][0]是一个长度为层数的列表，每个元素是size为[1,28,N,N]的tensor
        将每层attention先按注意力头平均再按层平均，最后得到NxN的矩阵"""
        # 将每层attention堆叠成[L, heads, N, N]
        attn_tensor = torch.zeros_like(attn[0][0].squeeze(0))
        for att in attn:
            attn_tensor+= att.squeeze(0)
        # 对head维度求平均 -> [L, N, N]
        head_avg = attn_tensor/len(attn)
        # 对layer维度求平均 -> [N, N]
        layer_avg = head_avg.mean(dim=0)
        return layer_avg

    def _compute_reward(self,obs_seq,processed_input,image_path,visualize,visual_save):
        """根据观察序列和这些text tokens关于image tokens的attention计算reward"""

        # 首先将obs 用batch的形式得到token ids
        obs_tag_seq = [tag for tag,_ in obs_seq]
        obs_range_seq,full_input_ids = self._get_obs_indices(processed_input,obs_seq)
        messages = self._make_messages(processed_input, image_path)
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
        input_ids = inputs["input_ids"]
        # 输入的整个input的组织形式应该如下所示：
        # <|im_start|>system
        # You are a helpful assistant.<|im_end|>
        # <|im_start|>user
        # <|vision_start|><|image_pad|><|vision_end|>
        # ......
        # <|im_end|>
        # <|im_start|>assistant

        vision_start = len(self.tokenizer(text.split("<|image_pad|>")[0], return_tensors='pt')["input_ids"][0])
        vision_end = vision_start+int(torch.prod(inputs["image_grid_thw"]) / (2 ** 2))

        outputs = self._chunk_forward(inputs, vision_start, vision_end,full_input_ids,self.chunk_size)

        obs_attn = outputs["attention"]
        obs_attn_seq = self._get_attn_seq(obs_attn,obs_range_seq,outputs)

        # 初始化history为均匀分布
        N = obs_attn_seq[0].size(0)
        history = torch.zeros_like(obs_attn_seq[0])

        shift_rewards = []  # 初始化shift_reward
        zoom_rewards = []  # 初始化zoom_reward
        for index, tag in enumerate(obs_tag_seq):

            cur_attn = obs_attn_seq[index]
            cur_attn_dis = self.uniform(cur_attn)

            # 首先，obs_seq每一个元素都是一段观察的文本+对应的tag(<zoom in>还是<shift>)
            if tag == 'shift':
                # 计算该段obs与history的KL散度
                if index == 0:
                    shift_reward, ratio_score, inst_ratio_score, contain_score = -1, 0, 0, 0
                else:
                    avg_history = history / index

                    comp_avg_history = avg_history.max() - avg_history
                    comp_avg_history_dis = self.uniform(comp_avg_history)
                    # visualize_tensor_distribution(comp_avg_history_dis)
                    # plot_1d_tensor(comp_avg_history_dis)
                    shift_reward, ratio_score, inst_ratio_score, contain_score = self._reward_func(comp_avg_history_dis,
                                                                                                   cur_attn_dis)

                shift_rewards.append(shift_reward)

            elif tag == 'zoom in':
                # 计算该段obs的attention与上一段obs的attention的包含度，即余弦相似度
                if index == 0:
                    zoom_reward, ratio_score, inst_ratio_score, contain_score = -1, 0, 0, 0
                else:
                    prev_attn = obs_attn_seq[index - 1]
                    prev_attn_dis = self.uniform(prev_attn)

                    zoom_reward, ratio_score, inst_ratio_score, contain_score = self._reward_func(prev_attn_dis,
                                                                                                  cur_attn_dis)

                zoom_rewards.append(zoom_reward)

            # 计算后，将该段obs的attention加入历史
            history += obs_attn_seq[index]

            # 是否需要可视化
            if visualize:
                heated_image = self.visualize(outputs,obs_attn_seq[index])
                if visual_save:
                    if not os.path.exists(visual_save):
                        os.mkdir(visual_save)
                    if tag == 'zoom in':
                        r = zoom_reward
                    cv2.imwrite(f'{visual_save}/{tag}_{index}_{r}.png',heated_image)
                    if tag == 'shift':
                        r = shift_reward
                    cv2.imwrite(f'{visual_save}/{tag}_{index}_{r}.png',heated_image)


        # log平缓
        shift_rewards,zoom_rewards = self._log_smooth(shift_rewards,zoom_rewards)
        return shift_rewards, zoom_rewards

    def split_tokens(self,all_tokens, max_len, vision_start,vision_end,full_input_ids):
        """
        分割文本 tokens
        """

        return [all_tokens[vision_end+i:vision_end+i+max_len] for i in range(0, len(full_input_ids), max_len)]

    def _chunk_forward(self, inputs,vision_start,vision_end,full_input_ids, max_text_len):

        # 获取文本 tokens
        all_tokens = inputs["input_ids"][0]

        # 切分文本 tokens
        text_chunks = self.split_tokens(all_tokens, max_text_len, vision_start, vision_end,full_input_ids)

        results = []
        for chunk in text_chunks:
            # 每次输入：图像 tokens + 当前文本片段 tokens
            chunk_input_ids = torch.cat([inputs["input_ids"][:, :vision_end], torch.tensor(chunk).unsqueeze(0)],
                                        dim=1)

            inputs_chunk = {
                "input_ids": chunk_input_ids,
                "pixel_values": inputs["pixel_values"],  # 保持图像输入不变
                "image_grid_thw": inputs['image_grid_thw'],
            }

            inputs_chunk = {k: v.to(self.model.device) for k, v in inputs_chunk.items()}

            with torch.inference_mode():
                outputs = self.model(
                    **inputs_chunk,
                    return_dict=True,
                    output_attentions=True
                )

            # 保存每个 chunk 的 attention
            results.append(outputs['attentions'])

        # 将所有 chunk 的 attention 合并
        # 最终的结果元组
        attn_result = []

        for idx in range(len(results[0])):
            # 收集所有元组中第 idx 个位置的 tensor
            tensors = [
                t[:, :, vision_end:, vision_start:vision_end]  # 切取最后一维
                for tup in results
                for j, t in enumerate(tup) if j == idx
            ]
            # 在 w 维度 (dim=2) 拼接
            merged = torch.cat(tensors, dim=2)
            attn_result.append(merged)

        del results
        # 转成元组
        attn_result = tuple(attn_result)
        all_attention = self._aggregate_attentions(attn_result)
        del attn_result
        torch.cuda.empty_cache()
        all_attention = all_attention[1:len(full_input_ids)+1,:]

        return {
            "attention": all_attention,  # 返回整个的 attention 矩阵
            "num_patches": int(torch.prod(inputs['image_grid_thw'],) / (2 ** 2)),
            "image_grid": inputs['image_grid_thw'],
        }

class ContainDisRewarder(Rewarder):
    def uniform(self,x):
        """将tensor转化成z-score分布"""

        def stretch_middle(x, gamma=2.0):
            x_min, x_max = x.min(), x.max()
            x_norm = (x - x_min) / (x_max - x_min)
            y = x_norm ** gamma
            return y * (x_max - x_min) + x_min

        def softmax_tensor(x):
            # 减去最大值防止数值溢出
            x_exp = torch.exp(x - torch.max(x))
            return x_exp / torch.sum(x_exp)

        def min_max_scale(x):
            return (x - x.min()) / (x.max() - x.min())

        def normalize(x):
            return x / x.sum()

        return normalize(min_max_scale(x))

    def instruction_func(self,score):
        if score < 0.5:
            return score
        else:
            return 1-score

    def _ratio_score(self,x,y):
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

    def _containing_score(self,x, y):
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
        comp_x = x.max()-x
        comp_x = comp_x / comp_x.sum()
        intersect = torch.min(comp_x, y).sum()
        union = x.sum()

        return 1 - (intersect / union)

    def _reward_func(self,x,y):
        ratio_score = self._ratio_score(x, y)
        inst_ratio_score = 2 * self.instruction_func(ratio_score)  # 乘以2是为了将值域缩放到[0,1]
        contain_score = self._containing_score(x, y)
        reward = inst_ratio_score * contain_score
        return reward, ratio_score, inst_ratio_score, contain_score

    def _log_smooth(self,shift_rewards,zoom_rewards):
        """log(动作的次数)*平均奖励"""
        shift_times_log = np.log(len(shift_rewards)+1)
        zoom_times_log = np.log(len(zoom_rewards)+1)
        shift_sum_ex = [p for p in shift_rewards if p != -1]
        zoom_sum_ex = [p for p in zoom_rewards if p != -1]
        shift_mean = np.mean(shift_sum_ex)
        zoom_mean = np.mean(zoom_sum_ex)
        shift_reward = shift_times_log * shift_mean
        zoom_reward = zoom_times_log * zoom_mean
        return shift_reward,zoom_reward
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

        shift_rewards = [] # 初始化shift_reward
        zoom_rewards = [] # 初始化zoom_reward
        for index, tag in enumerate(obs_tag_seq):

            cur_attn = obs_attn_seq[index]
            cur_attn_dis = self.uniform(cur_attn)

            # 首先，obs_seq每一个元素都是一段观察的文本+对应的tag(<zoom in>还是<shift>)
            if tag == 'shift':
                # 计算该段obs与history的KL散度
                if index == 0:
                    shift_reward, ratio_score, inst_ratio_score, contain_score = -1,0,0,0
                else:
                    avg_history = history / index

                    comp_avg_history = avg_history.max() - avg_history
                    comp_avg_history_dis = self.uniform(comp_avg_history)
                    # visualize_tensor_distribution(comp_avg_history_dis)
                    # plot_1d_tensor(comp_avg_history_dis)
                    shift_reward, ratio_score, inst_ratio_score, contain_score = self._reward_func(comp_avg_history_dis, cur_attn_dis)

                shift_rewards.append(shift_reward)

            elif tag == 'zoom in':
                # 计算该段obs的attention与上一段obs的attention的包含度，即余弦相似度
                if index == 0:
                    zoom_reward, ratio_score, inst_ratio_score, contain_score = -1, 0, 0, 0
                else:
                    prev_attn = obs_attn_seq[index - 1]
                    prev_attn_dis = self.uniform(prev_attn)

                    zoom_reward, ratio_score, inst_ratio_score, contain_score = self._reward_func(prev_attn_dis, cur_attn_dis)

                zoom_rewards.append(zoom_reward)

            # 计算后，将该段obs的attention加入历史
            history += obs_attn_seq[index]

            # 是否需要可视化
            if visualize:
                heated_image = self.visualize(outputs, obs_attn_seq[index])
                if visual_save:
                    if not os.path.exists(visual_save):
                        os.mkdir(visual_save)
                    if tag == 'zoom in':
                        r = zoom_reward
                        contain_str = f"{contain_score:.3g}"
                        ratio_str = f"{ratio_score:.3g}"
                        inst_ratio_score_str = f"{inst_ratio_score:.3g}"
                        cv2.imwrite(f'{visual_save}/{tag}_{index}_{r:.3g}_c{contain_str}_ir{inst_ratio_score_str}_r{ratio_str}.png', heated_image)
                    if tag == 'shift':
                        r = shift_reward
                        contain_str = f"{contain_score:.3g}"
                        ratio_str = f"{ratio_score:.3g}"
                        cv2.imwrite(f'{visual_save}/{tag}_{index}_{r:.3g}_c{contain_str}_ir{inst_ratio_score_str}_r{ratio_str}.png', heated_image)

        # log（次数）* 平均的奖励
        shift_rewards, zoom_rewards = self._log_smooth(shift_rewards, zoom_rewards)
        return shift_rewards, zoom_rewards

class ContainRewarder(ContainDisRewarder):
    def _compute_reward(self, obs_seq, processed_input, outputs, visualize, visual_save):
        """根据观察序列和这些text tokens关于image tokens的attention计算reward"""

        attn = outputs['aggregate_attn']
        num_patches = outputs['num_patches']
        processed_image = outputs['processed_image']
        # 首先将obs 用batch的形式得到token ids
        obs_tag_seq = [tag for tag, _ in obs_seq]
        obs_range_seq, full_input_ids = self._get_obs_indices(processed_input, obs_seq)

        # 输入的整个input的组织形式应该如下所示：
        # <|im_start|>system
        # You are a helpful assistant.<|im_end|>
        # <|im_start|>user
        # <|vision_start|><|image_pad|><|vision_end|>
        # ......
        # <|im_end|>
        # <|im_start|>assistant

        vision_start = len(
            self.tokenizer(outputs['text'].split("<|image_pad|>")[0], return_tensors='pt')["input_ids"][0])
        vision_end = vision_start + num_patches
        obs_attn = attn[vision_end + 1:vision_end + 1 + len(full_input_ids),
                   vision_start:vision_end]  # obs_attn的形状应该是 obs文本的长度*图片tokens的长度
        obs_attn_seq = self._get_attn_seq(obs_attn, obs_range_seq, outputs)

        # 初始化history为均匀分布
        N = obs_attn_seq[0].size(0)
        history = torch.zeros_like(obs_attn_seq[0])

        shift_rewards = 0  # 初始化shift_reward
        zoom_rewards = 0  # 初始化zoom_reward
        for index, tag in enumerate(obs_tag_seq):

            cur_attn = obs_attn_seq[index]

            # 首先，obs_seq每一个元素都是一段观察的文本+对应的tag(<zoom in>还是<shift>)
            if tag == 'shift':
                # 计算该段obs与history的KL散度
                if index == 0:
                    shift_reward, ratio_score, inst_ratio_score, contain_score = 0,0,0,0
                else:
                    avg_history = history / index

                    comp_avg_history = avg_history.max() - avg_history
                    ratio_score = self._ratio_score(comp_avg_history, cur_attn)
                    inst_ratio_score = 2 * self.instruction_func(ratio_score)  # 乘以2是为了将值域缩放到[0,1]
                    contain_score = self._containing_score(comp_avg_history,cur_attn)
                    shift_reward = contain_score * inst_ratio_score

                shift_rewards += shift_reward

            elif tag == 'zoom in':
                # 计算该段obs的attention与上一段obs的attention的包含度，即余弦相似度
                if index == 0:
                    zoom_reward, ratio_score, inst_ratio_score, contain_score = 0, 0, 0, 0
                else:
                    prev_attn = obs_attn_seq[index - 1]
                    ratio_score = self._ratio_score(prev_attn, cur_attn)
                    inst_ratio_score = 2 * self.instruction_func(ratio_score)  # 乘以2是为了将值域缩放到[0,1]
                    contain_score = self._containing_score(prev_attn, cur_attn)

                    zoom_reward = contain_score * inst_ratio_score

                zoom_rewards += zoom_reward

            # 计算后，将该段obs的attention加入历史
            history += obs_attn_seq[index]

            # 是否需要可视化
            if visualize:
                heated_image = self.visualize(outputs, obs_attn_seq[index])
                if visual_save:
                    if not os.path.exists(visual_save):
                        os.mkdir(visual_save)
                    if tag == 'zoom in':
                        r = zoom_reward
                        contain_str = f"{contain_score:.3g}"
                        ratio_str = f"{ratio_score:.3g}"
                        cv2.imwrite(f'{visual_save}/{tag}_{index}_{r:.3g}_c{contain_str}_r{ratio_str}.png', heated_image)
                    if tag == 'shift':
                        r = shift_reward
                        contain_str = f"{contain_score:.3g}"
                        ratio_str = f"{ratio_score:.3g}"
                        cv2.imwrite(f'{visual_save}/{tag}_{index}_{r:.3g}_c{contain_str}_r{ratio_str}.png', heated_image)

        # log平缓
        shift_rewards, zoom_rewards = self._log_smooth(shift_rewards, zoom_rewards)
        return shift_rewards, zoom_rewards

if __name__ == "__main__":
    import sys
    
    # 检查命令行参数以决定是否使用 accelerate
    use_accelerate = False
    
    if use_accelerate:
        print("使用 Accelerate 多卡模式...")
        # 示例：使用不同类型的 rewarder
        # rewarder = Rewarder(model_path="/data/pretrained_models/Qwen2.5-VL-7B-Instruct", use_accelerate=True)
        # rewarder = ChunkRewarder(chunk_size=7200, model_path="/data/pretrained_models/Qwen2.5-VL-7B-Instruct", use_accelerate=True)
        rewarder = ContainDisRewarder(model_path="/data/pretrained_models/Qwen2.5-VL-7B-Instruct", use_accelerate=True)
    else:
        print("使用单卡模式...")
        rewarder = ChunkRewarder(chunk_size=2000,model_path="/data/pretrained_models/Qwen2.5-VL-7B-Instruct")
        # rewarder = Rewarder(model_path="/data/pretrained_models/Qwen2.5-VL-7B-Instruct")
        # rewarder = ContainDisRewarder(model_path="/data/pretrained_models/Qwen2.5-VL-7B-Instruct")
    e_text = f"""Let's observe step by step.
<zoom in>
The image shows a web page with a list of listings in the 'Rvs + campers' category. The listings are sorted by 'Newly listed.' The current focus is on the title "Rvs + campers," which is clickable to further refine the search.
</zoom in>
<shift>
To narrow down the search to a specific game category, such as the Steam Workshop, the next logical step is to click on the category search bar to input the desired game, allowing the system to filter results accordingly.
</shift>
<zoom in>
The "Rvs + campers" category is visible and should be interacted with to filter the listings appropriately.
</zoom in>
<shift>
Clicking on the 'Rvs + campers' category should allow further refinement of the listings to find the most recently listed item related to RAM.
</shift>
<zoom in>
The observed interface suggests a need to interact with the list of categories or specific listings to filter or refine the search, which will facilitate finding the RAM details.
</zoom in>
<shift>
To achieve the task, we should click on the 'Rvs + campers' category as it is the current focus.
</shift>
<zoom in>
The "Rvs + campers" category is the key focus, and interacting with it should allow further refinement and access to relevant listings.
</zoom in>
<shift>
Click on the 'Rvs + campers' category to filter the listings and refine the search to find the most recently listed item.
</shift>
<zoom in>
To find the most recently listed item, we need to filter the listings to focus on 'Video gaming' instead of 'Rvs + campers.' Clicking on the 'Video gaming' category will refine results accordingly.
</zoom in>
<shift>
Click on the 'Video gaming' category in the refine section to filter the listings.
</shift>
<summary>
Click on the 'Rvs + campers' category to interact with it and verify or refine the filters for the listings.
```
click [14]
```</summary>"""

    bug_text = """Let's observe step by step. First, I will zoom in to the region where keyword and other needed elements are located.
<zoom in>
The current page includes:
- Keyword search box, which can receive a specific search query to refine results.
- A Category selection dropdown marked by id 6.
- A Price section denoted by id 8 which includes sliders for price range refinement.
</zoom in>
I will need to shift focus to the correct section to set the criteria specific to find a red Toyota within the given price range.
<shift>
After identifying the elements on the page, I should:
1. Click on the Category dropdown (id 6) to explore subcategories.
2. Click the Price Min. and Max. sliders' area (id 8) to set the price range $3000 to $6000.
</shift>
Following these steps:
1. I need to set the category to 'Cars & Trucks'.
2. I will then narrow the price range to $3000 to $6000.
3. After this, I will perform a search.

The next action should be clicking on the 'Cars + Trucks' under "All categories".
<action>
```click [39]```
</action>
<summary>
The next steps would be searching the keyword 'Toyota' within the 'Cars + Trucks' category, setting the price range to $3000-$6000, and executing the search. Then, once focusing on the Toyota listings within this range, a link would be clicked to view the page of the cheapest red Toyota. The action performed here is choosing the 'Cars + Trucks' category.
</summary>
<tool_call>

 addCriterion
<tool_call>






 addCriterion

 addCriterion

 addCriterion


















































































































































"""
    image_path = "/data/wangzhenchuan/Projects/LIFT/results/55/step_3_obs.png"

    print(rewarder.reward(bug_text, visualize=False, image_path=image_path,
                          visual_save='/data/wangzhenchuan/Projects/LIFT/visualize_debug_low'))
    e_text = """Let's observe step-by-step. First, I will zoom in to observe the whole page.
<zoom in>
This page can be divided into following sections:
**Header Section**:
- Contains "OsClass" logo and navigation links for "My account," "Logout," and "Publish Ad."

**Navigation Bar**:
- Includes options like "Classifieds" > Furniture.

**Search Filters Area**:
- Allows users to input search terms, select cities, show only listings with pictures, set price ranges using sliders or text fields.

**Subscribe Box**:
- Option allowing visitors to subscribe via email notifications about new furniture items matching their criteria; includes an orange box labeled “Subscribe now!”

**Refine Category Options**:
- Links allow narrowing down searches further: all categories and 'Furniture'.

**Main Content Area – Listings Displayed** :
- Lists available furniture products including images, titles, prices, locations alongwith brief descriptions below.
</zoom in>
According to the observation above, this page already displays products of "Furniture" category and the products are already \
sorted by dates. To find the most recent blue chair in the "Furniture" category of Washington, D.C. I should zoom in the Main Content Area \
to see if there is any blue chair.
<zoom in>
The "Main Content Area" section of this page displays three furniture items for sale:

### Century Furniture English Roll Arm Sofa
- **Title:** Century Furniture English Roll Arm Sofa
- **Price:** $605.00 ($23 less than original)
- **Location:** Arlington, Virginia / Added: November 16th, 2023

**Description:**
```
SAVE UP TO 90%! PRICES UPDATED DAILY!
Century Furniture English Roll Arm sofa.
Original Price was $5000 now only at $605.00
Brand: Century Furniture
```

---

### Highland House Tufted Back Accent Chair
- **Title:** Highland House Furniture Tufted Back Accent Chair
- **Price:** $220.00 ($78 off from Original)
- **Location:** Dale City, Virginia / Added: November 16th, 2023

**Description:**
```
SAVE UP TO 90% !PRICES UPDATED DAILY!
Highland house furniture tufted back accent chair.
original price :$4500 now it's just:$220.00
brand high land house furniture
```

---

### NEW Zinus Green Tea QUEEN Memory Foam Mattress
- **Title:** New Zinus 12 Inch Green Tea Queen Memory Foam Mattress
- **Price:** $199.00 ($401 discount compared to retail value)

- **Location:** Borough of East Washington, Pennsylvania / Added: November 16th, 2023

**Description:**
```
Zinus 12 Inch green tea queen memory foam mattress certipur-us certified bed-in-a-box pressure relieving queen.
This bed retails for $600 get it for one-third its price of $199 !
Also have bedframes available at steep discounts if you want to save...
```
</zoom in>
According to the observation above, only Highland House Tufted Back Accent Chair is a chair in Washington, D.C., I need to zoom in its thumbnail image \
to see if it's blue.
<zoom in>
Focusing on the thumbnail for the Highland House Tufted Back Accent Chair, the chair’s upholstery is a \
light beige/cream color with deep button tufting—not blue.
</zoom in>
According to the observation above, none of the products in this page is the most recent blue chair in Washington, D.C. \
So I need to shift to Search Filters Area to narrow down the products displayed to products in Washington, D.C.
<shift>
We move our focus to the **Search Filters Area** on the left:
The Search Filters Area is located on the left side of the webpage and contains several options to refine search results:

1. **Your search**: A text box where users can enter specific keywords or phrases related to their furniture needs.
2. **City**: Another input field for specifying the city in which they want to find listings, allowing searches within particular geographic areas.
3. **Show only listings with pictures**: An option that filters out ads without images if selected by checking this checkbox (not checked here).
4. **Price Min./Max.:**
   - Two fields labeled "Min." and "Max.", enabling price range filtering so you specify your budget limits when searching.

5. **Apply button:** This blue rectangular button allows applying any changes made through these filter settings back into the main listing area above it after entering values like prices etc., making sure all criteria match before displaying relevant items accordingly based upon those inputs provided earlier via respective dropdown menus/checkboxes available under each section mentioned previously herein above.
</shift>
According to the observation above, I can input "Washington" in **City** to narrow down displayed products. Next, I need \
to zoom in to check the id of **City**.
<zoom in>
The id of **City** is 7
</zoom in>

<summary>
Observations so far:
1. The page is divided into a few sections.
2. The Main Content displays three items added November 16, 2023:
   - Century Furniture English Roll Arm Sofa (Arlington, VA)
   - Highland House Tufted Back Accent Chair (Dale City, VA)
   - Zinus Green Tea Queen Mattress (East Washington, PA)
3. Zooming the Highland House Accent Chair thumbnail shows it is beige, not blue.
4. No blue chair in Washington, D.C. appears on this page, so we need to use the Search Filters Area to narrow results by city.

So the next action I will perform is ```type [7] [Washington] [0]```
</summary>"""
    image_path = "/data/wangzhenchuan/Projects/LIFT/data/example/example.png"

    print(rewarder.reward(e_text,visualize=False,image_path=image_path, visual_save = '/data/wangzhenchuan/Projects/LIFT/visualize_debug'))

    # A = torch.tensor(
    #     [[1, 0, 0, 0],
    #      [0, 0, 1, 1],
    #      [0, 1, 1, 1],
    #      [0, 0, 0, 0]])
    # A = A.max()- A
    # A = A / A.sum()
    # B = torch.tensor(
    #     [[0, 1, 1, 1],
    #      [1, 1, 0, 0],
    #      [1, 0, 0, 0],
    #      [1, 0, 0, 0]])
    # B = B / B.sum()
    #
    # ratio_score = 2*rewarder.instruction_func(rewarder._ratio_score(A,B))
    # contain_score = rewarder._containing_score(A,B)
    # reward = contain_score * ratio_score
    # print(reward)