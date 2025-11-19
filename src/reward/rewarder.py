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
from src.reward.reward_tools import format_reward_cal

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
        return format_reward_cal(response)

    def reward(self,response,image_path,visualize = False,visual_save = None,visualize_per_token = False,visualize_obs_indices = None):
        """
        response: 模型输出的text tokens
        image: 对应的环境的截图
        visualize_per_token: 是否生成每个token的可视化（包括热力图、柱状图和交互式HTML）
        visualize_obs_indices: 指定要可视化的观察序列索引列表，如 [0, 2, 5]。如果为 None，则可视化所有观察序列
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

        format_reward = self._compute_format_reward(response)

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
        shift_reward,zoom_reward = self._compute_reward(obs_seq,processed_input,outputs,visualize,visual_save,visualize_per_token,visualize_obs_indices)
        # query peak GPU memory
        if torch.cuda.is_available() and device.type == 'cuda':
            peak_bytes = torch.cuda.max_memory_allocated(device)
            peak_mib = peak_bytes / (1024 ** 2)
            print(f"[Rewarder] Peak GPU memory during reward(): {peak_mib:.1f} MiB")

        return shift_reward,zoom_reward,format_reward

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

    def _compute_reward(self,obs_seq,processed_input,outputs,visualize,visual_save,visualize_per_token=False,visualize_obs_indices=None):
        """根据观察序列和这些text tokens关于image tokens的attention计算reward

        visualize_obs_indices: 指定要可视化的观察序列索引列表，如果为 None 则可视化所有
        """

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

            # 是否需要per-token可视化
            if visualize_per_token and visual_save:
                # 检查是否需要可视化当前观察序列
                if visualize_obs_indices is None or index in visualize_obs_indices:
                    st, ed, slice_ids = obs_range_seq[index]
                    self.visualize_per_token_attention(
                        obs_attn=obs_attn[st:ed],
                        token_ids=slice_ids,
                        token_start_idx=st,  # 添加起始索引
                        vision_start=vision_start,
                        vision_end=vision_end,
                        tag=tag,
                        obs_idx=index,
                        visual_save=visual_save,
                        outputs=outputs,
                        full_attn=attn,
                        obs_range_seq=obs_range_seq,  # 传入所有观察序列的范围
                        full_input_ids=full_input_ids  # 新增：传入完整的input_ids
                    )


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

    def visualize_per_token_attention(self, obs_attn, token_ids, token_start_idx, vision_start, vision_end,
                                       tag, obs_idx, visual_save, outputs, full_attn, obs_range_seq, full_input_ids):
        """
        为观察序列中的每个token生成详细的attention可视化

        参数:
            obs_attn: [N_tokens, N_image_patches] 当前观察序列的attention矩阵
            token_ids: 当前观察序列的token IDs列表
            token_start_idx: 当前观察序列在full_input_ids中的起始索引
            vision_start: 图片tokens的起始位置
            vision_end: 图片tokens的结束位置
            tag: 动作标签 ('shift' 或 'zoom in')
            obs_idx: 观察序列索引
            visual_save: 可视化保存路径
            outputs: 模型输出字典
            full_attn: [N, N] 完整的attention矩阵
            obs_range_seq: 所有观察序列的token范围列表
            full_input_ids: 完整的观察序列token IDs（包括所有观察序列和gap）
        """
        import matplotlib.pyplot as plt

        if not os.path.exists(visual_save):
            os.makedirs(visual_save)

        # 为每个token生成可视化
        token_data = []
        for token_idx in range(len(token_ids)):
            token_id = token_ids[token_idx]
            token_text = self.tokenizer.decode([token_id])

            # 获取当前token在完整attn中的位置（使用正确的全局索引）
            global_token_idx = vision_end + 1 + token_start_idx + token_idx

            # 提取当前token对所有上下文的attention
            # full_attn[global_token_idx, :] 包含对所有token的attention
            token_full_attn = full_attn[global_token_idx, :global_token_idx+1]  # 只看之前的token

            # 分离图片和文本部分
            token_attn_to_image = token_full_attn[vision_start:vision_end]  # 对图片的attention

            # 对文本的attention：只包含观察序列的文本（不含系统提示）
            # vision_end+1 到当前token之前的所有观察文本
            token_attn_to_text = token_full_attn[vision_end+1:-1]

            # 收集所有前文观察token的文本（用于可视化x轴）
            # 直接从 full_input_ids 中提取从0到当前token之前的所有tokens（包括gap）
            preceding_token_texts = []

            # 计算需要的总token数：当前观察序列的起始位置 + 当前token在该序列中的偏移
            total_preceding_count = token_start_idx + token_idx

            # 从 full_input_ids 中提取所有前文tokens
            for i in range(total_preceding_count):
                tid = full_input_ids[i]
                preceding_token_texts.append(self.tokenizer.decode([tid]))

            token_data.append({
                'token_idx': token_idx,
                'token_id': token_id,
                'token_text': token_text,
                'attn_to_image': token_attn_to_image.cpu().numpy(),
                'attn_to_text': token_attn_to_text.cpu().numpy() if len(token_attn_to_text) > 0 else None,
                'full_attn': token_full_attn.cpu().numpy(),
                'preceding_token_texts': preceding_token_texts  # 新增：前文token的文本
            })

            # 生成矩阵热力图
            # self._generate_token_heatmap(
            #     token_data[-1],
            #     f'{visual_save}/{tag}_{obs_idx}_token_{token_idx}_{token_text[:20]}_heatmap.png'
            # )

            # 生成柱状图
            self._generate_token_barchart(
                token_data[-1],
                f'{visual_save}/{tag}_{obs_idx}_token_{token_idx}_{token_text[:20]}_bar.png',
                outputs
            )

            # 生成HTML表格可视化（更清晰地展示每个前文token的attention）
            self._generate_token_text_attention_html(
                token_data[-1],
                f'{visual_save}/{tag}_{obs_idx}_token_{token_idx}_{token_text[:20]}_text_attn.html'
            )

        # 生成交互式HTML总览
        # self._generate_interactive_html(
        #     token_data,
        #     tag,
        #     obs_idx,
        #     f'{visual_save}/{tag}_{obs_idx}_per_token_overview.html'
        # )

    def _generate_token_heatmap(self, token_data, save_path):
        """
        生成单个token的attention矩阵热力图

        参数:
            token_data: 包含token信息和attention数据的字典
            save_path: 保存路径
        """
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns
            use_seaborn = True
        except ImportError:
            use_seaborn = False

        fig, ax = plt.subplots(figsize=(12, 2))

        # 将full_attn重塑为2D以便可视化
        attn_2d = token_data['full_attn'].reshape(1, -1)

        if use_seaborn:
            import seaborn as sns
            sns.heatmap(attn_2d, ax=ax, cmap='viridis', cbar=True,
                        xticklabels=False, yticklabels=False)
        else:
            # 使用纯 matplotlib
            im = ax.imshow(attn_2d, cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_title(f"Token: '{token_data['token_text']}' - Attention Distribution")
        ax.set_xlabel('Context Position')
        ax.set_ylabel('Current Token')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _generate_token_barchart(self, token_data, save_path, outputs):
        """
        生成单个token的attention柱状图，区分图片和文本部分

        参数:
            token_data: 包含token信息和attention数据的字典
            save_path: 保存路径
            outputs: 模型输出字典（包含图片网格信息）
        """
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        # 左图：对图片patches的attention
        attn_to_image = token_data['attn_to_image']
        ax1.bar(range(len(attn_to_image)), attn_to_image, color='steelblue', alpha=0.7)
        ax1.set_title(f"Token '{token_data['token_text']}' - Attention to Image Patches")
        ax1.set_xlabel('Image Patch Index')
        ax1.set_ylabel('Attention Weight')
        ax1.grid(axis='y', alpha=0.3)

        # 右图：对文本tokens的attention
        if token_data['attn_to_text'] is not None and len(token_data['attn_to_text']) > 0:
            attn_to_text = token_data['attn_to_text']
            token_texts = token_data['preceding_token_texts']

            # 确保长度匹配（调试和修正）
            if len(token_texts) != len(attn_to_text):
                print(f"[Warning] Length mismatch for token '{token_data['token_text']}' (idx={token_data['token_idx']})")
                print(f"  attn_to_text length: {len(attn_to_text)}")
                print(f"  token_texts length: {len(token_texts)}")

                # 截断到较短的长度
                min_len = min(len(token_texts), len(attn_to_text))
                token_texts = token_texts[:min_len]
                attn_to_text = attn_to_text[:min_len]
                print(f"  Truncated to length: {min_len}")

            # 使用柱状图，x轴为token文本
            x_positions = range(len(attn_to_text))
            ax2.bar(x_positions, attn_to_text, color='coral', alpha=0.7)
            ax2.set_xticks(x_positions)
            ax2.set_xticklabels(token_texts, rotation=45, ha='right', fontsize=8)
            ax2.set_title(f"Token '{token_data['token_text']}' - Attention to Preceding Tokens")
            ax2.set_xlabel('Preceding Token Text')
            ax2.set_ylabel('Attention Weight')
            ax2.grid(axis='y', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No preceding text tokens',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Attention to Preceding Tokens')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _generate_token_text_attention_html(self, token_data, save_path):
        """
        生成单个token对前文tokens的attention的HTML表格可视化

        特点：
        - 每行显示一个前文token和attention值
        - 根据attention值大小使用颜色编码
        - 包含水平柱状条形图
        - 支持按attention值排序

        参数:
            token_data: 包含token信息和attention数据的字典
            save_path: HTML文件保存路径
        """
        import numpy as np

        # 获取数据
        current_token = token_data['token_text']
        token_idx = token_data['token_idx']
        attn_to_text = token_data['attn_to_text']
        preceding_token_texts = token_data['preceding_token_texts']

        # 如果没有前文文本，返回空文件
        if attn_to_text is None or len(attn_to_text) == 0:
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Token '{current_token}' - No Preceding Tokens</title>
</head>
<body>
    <h2>Token '{current_token}' (Index: {token_idx})</h2>
    <p>No preceding text tokens available.</p>
</body>
</html>
"""
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return

        # 确保长度匹配
        if len(preceding_token_texts) != len(attn_to_text):
            min_len = min(len(preceding_token_texts), len(attn_to_text))
            preceding_token_texts = preceding_token_texts[:min_len]
            attn_to_text = attn_to_text[:min_len]

        # 归一化attention值到0-1范围用于可视化
        attn_values = np.array(attn_to_text)
        max_attn = attn_values.max() if attn_values.max() > 0 else 1.0
        normalized_attn = attn_values / max_attn

        # 生成表格行
        table_rows = []
        for i, (token_text, attn_val, norm_attn) in enumerate(zip(preceding_token_texts, attn_values, normalized_attn)):
            # 根据attention值决定背景颜色
            if norm_attn > 0.7:
                row_class = 'high-attn'
            elif norm_attn > 0.4:
                row_class = 'medium-attn'
            else:
                row_class = 'low-attn'

            # 转义HTML特殊字符
            token_display = token_text.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
            if token_display.strip() == '':
                token_display = '[SPACE/NEWLINE]'

            # 柱状图宽度（最大500px）
            bar_width = int(norm_attn * 500)

            row_html = f"""
                <tr class="{row_class}" data-attn="{attn_val:.6f}">
                    <td>{i}</td>
                    <td class="token-cell">{token_display}</td>
                    <td class="attn-value">{attn_val:.6f}</td>
                    <td class="bar-cell">
                        <div class="bar" style="width: {bar_width}px;"></div>
                    </td>
                </tr>
"""
            table_rows.append(row_html)

        # 构建完整HTML
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Token '{current_token}' - Attention to Preceding Tokens</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            max-width: 1200px;
            margin: 0 auto;
        }}
        h2 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .info {{
            margin-bottom: 20px;
            padding: 10px;
            background-color: #e8f4f8;
            border-left: 4px solid #2196F3;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-top: 20px;
        }}
        th, td {{
            padding: 10px;
            border: 1px solid #ddd;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            cursor: pointer;
            user-select: none;
        }}
        th:hover {{
            background-color: #45a049;
        }}
        .token-cell {{
            font-family: 'Courier New', monospace;
            font-weight: bold;
            white-space: pre;
        }}
        .attn-value {{
            font-family: 'Courier New', monospace;
            text-align: right;
        }}
        .bar-cell {{
            min-width: 500px;
        }}
        .bar {{
            height: 20px;
            background: linear-gradient(90deg, #4CAF50, #2196F3);
            border-radius: 3px;
            transition: width 0.3s;
        }}
        .high-attn {{
            background-color: rgba(255, 100, 100, 0.2);
        }}
        .medium-attn {{
            background-color: rgba(255, 200, 100, 0.2);
        }}
        .low-attn {{
            background-color: rgba(200, 200, 200, 0.1);
        }}
        tr:hover {{
            background-color: rgba(33, 150, 243, 0.1) !important;
        }}
        .controls {{
            margin: 15px 0;
        }}
        button {{
            padding: 8px 16px;
            margin-right: 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        button:hover {{
            background-color: #45a049;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h2>Token: '{current_token}' (Index: {token_idx})</h2>
        <div class="info">
            <strong>Total Preceding Tokens:</strong> {len(attn_values)} |
            <strong>Max Attention:</strong> {attn_values.max():.6f} |
            <strong>Mean Attention:</strong> {attn_values.mean():.6f}
        </div>

        <div class="controls">
            <button onclick="sortTable(2, false)">Sort by Attention (Desc)</button>
            <button onclick="sortTable(0, true)">Sort by Index (Asc)</button>
        </div>

        <table id="attnTable">
            <thead>
                <tr>
                    <th onclick="sortTable(0, true)">Index ▲</th>
                    <th onclick="sortTable(1, true)">Token</th>
                    <th onclick="sortTable(2, false)">Attention ▼</th>
                    <th>Visualization</th>
                </tr>
            </thead>
            <tbody>
{''.join(table_rows)}
            </tbody>
        </table>
    </div>

    <script>
        let sortOrder = {{}};

        function sortTable(columnIndex, ascending = true) {{
            const table = document.getElementById('attnTable');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));

            // Toggle sort order if clicking same column
            if (sortOrder[columnIndex] !== undefined) {{
                ascending = !sortOrder[columnIndex];
            }}
            sortOrder = {{}};
            sortOrder[columnIndex] = ascending;

            rows.sort((a, b) => {{
                let aVal, bVal;

                if (columnIndex === 0) {{
                    // Index column
                    aVal = parseInt(a.cells[0].textContent);
                    bVal = parseInt(b.cells[0].textContent);
                }} else if (columnIndex === 2) {{
                    // Attention column
                    aVal = parseFloat(a.dataset.attn);
                    bVal = parseFloat(b.dataset.attn);
                }} else {{
                    // Token column
                    aVal = a.cells[1].textContent;
                    bVal = b.cells[1].textContent;
                }}

                if (aVal < bVal) return ascending ? -1 : 1;
                if (aVal > bVal) return ascending ? 1 : -1;
                return 0;
            }});

            // Re-append sorted rows
            rows.forEach(row => tbody.appendChild(row));
        }}
    </script>
</body>
</html>
"""

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _generate_interactive_html(self, token_data_list, tag, obs_idx, save_path):
        """
        生成交互式HTML可视化，展示整段观察的所有token

        参数:
            token_data_list: 包含所有token数据的列表
            tag: 动作标签
            obs_idx: 观察序列索引
            save_path: HTML保存路径
        """
        import json

        # 准备数据
        tokens = [d['token_text'] for d in token_data_list]

        # 构建HTML
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{tag} - Observation {obs_idx} - Per-Token Attention</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
        }}
        .container {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        #heatmap {{
            width: 100%;
            height: 600px;
        }}
        .info {{
            margin-top: 20px;
            padding: 15px;
            background-color: #e8f4f8;
            border-left: 4px solid #2196F3;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Per-Token Attention Visualization</h1>
        <div class="info">
            <strong>Tag:</strong> {tag} |
            <strong>Observation Index:</strong> {obs_idx} |
            <strong>Total Tokens:</strong> {len(tokens)}
        </div>
        <div id="heatmap"></div>
    </div>

    <script>
        var tokens = {json.dumps(tokens)};
        var attentions = {json.dumps([d['full_attn'].tolist() for d in token_data_list])};

        // 创建热力图数据
        var data = [{{
            z: attentions,
            x: Array.from({{length: Math.max(...attentions.map(a => a.length))}}, (_, i) => i),
            y: tokens,
            type: 'heatmap',
            colorscale: 'Viridis',
            hoverongaps: false,
            hovertemplate: 'Token: %{{y}}<br>Context Position: %{{x}}<br>Attention: %{{z:.4f}}<extra></extra>'
        }}];

        var layout = {{
            title: 'Token-wise Attention Distribution',
            xaxis: {{
                title: 'Context Position (0 = start)',
                side: 'bottom'
            }},
            yaxis: {{
                title: 'Token',
                autorange: 'reversed'
            }},
            height: 600
        }};

        Plotly.newPlot('heatmap', data, layout, {{responsive: true}});
    </script>
</body>
</html>
"""

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

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

        format_reward = self._compute_format_reward(response)


        # 1. 从response中提取标签,组装新的input_text,并提取出观察序列
        processed_input,obs_seq = self._get_processed_input(response)

        if len(obs_seq)== 0:
            # 如果没有观察序列，可以直接返回了
            return 0,0,0.025*format_reward

        # 4. 计算reward
        shift_reward,zoom_reward = self._compute_reward(obs_seq,processed_input,image_path,visualize,visual_save)
        # query peak GPU memory
        if torch.cuda.is_available() and device.type == 'cuda':
            peak_bytes = torch.cuda.max_memory_allocated(device)
            peak_mib = peak_bytes / (1024 ** 2)
            print(f"[Rewarder] Peak GPU memory during reward(): {peak_mib:.1f} MiB")

        return shift_reward,zoom_reward,0.025*format_reward

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
    image_path = "/data/wangzhenchuan/Projects/LIFT/data/debug/visualize_guitar.png"
    e_text = """Let's break down the task and focused on the page interactions.
<zoom in>
The user's objective is to find the email of the seller of the guitar in the red case on the current page. The screenshot currently displays a list of musical instruments, and there are specific listings to consider, particularly focusing on guitars.

Observations:
- The current page is listing music instruments, including guitars.
- The listing highlighted in red appears to be for a "2021 GIBSON SG TRIBUTE with GIBSON hard case."
- To find the email of the seller, we need to click on the relevant listing to view more details, including the seller's contact information.

The next step is to specifically locate the listing for the "2021 GIBSON SG TRIBUTE with GIBSON hard case." Instead of providing the email directly, it would be necessary to click on the listing to proceed to the seller's contact page. Therefore, any relevant listing must be clicked for detailed information.
</zoom in>

<action>
click [32]
</action>"""
    if use_accelerate:
        print("使用 Accelerate 多卡模式...")
        # 示例：使用不同类型的 rewarder
        # rewarder = Rewarder(model_path="/data/pretrained_models/Qwen2.5-VL-7B-Instruct", use_accelerate=True)
        # rewarder = ChunkRewarder(chunk_size=7200, model_path="/data/pretrained_models/Qwen2.5-VL-7B-Instruct", use_accelerate=True)
        rewarder = ContainDisRewarder(model_path="/data/pretrained_models/Qwen2.5-VL-7B-Instruct", use_accelerate=True)
    else:
        print("使用单卡模式...")
        # rewarder = ChunkRewarder(chunk_size=2000,model_path="/data/pretrained_models/Qwen2.5-VL-7B-Instruct")
        rewarder = Rewarder(model_path="/data/pretrained_models/Qwen2.5-VL-7B-Instruct")
        # rewarder = ContainDisRewarder(model_path="/data/pretrained_models/Qwen2.5-VL-7B-Instruct")

    print(rewarder.reward(e_text,
                          image_path=image_path,
                          visualize=True,
                          visual_save='/data/wangzhenchuan/Projects/LIFT/visualize_attention_guitar_good',
                          visualize_per_token=True,
                          visualize_obs_indices = [0]))
