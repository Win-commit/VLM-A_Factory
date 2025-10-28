import pathlib
import numpy as np
import json
from torch.utils.data import Dataset
from typing import List, Dict, Any
from PIL import Image

LIBERO_RAW_DATA_PATH = pathlib.Path('/liujinxin/zhy/ICLR2026/datasets/libero/data/libero_all')

SYSTEM_PROMPT = "Analyze the input image and predict robot actions for the next {H} timesteps. Each action has {D} dimensions. Output a single sequence of {H}*{D} integers (0-{B} each), representing the {H} timesteps sequentially. Provide only space-separated numbers. Nothing else."



'''
TODO: 把libero数据集处理成经典llm微调的格式,任意时刻t0应该对应于字符串输入:
message: [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": main_image_files
            },
            {
                "type": "image",
                "image": gripper_image_files
            },
            {
                "type": "text",
                "text": task_instruction
            }
        ]
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": action_files(H*D个范围在[0,B]的整数，归一化得到)
            }
        ]
    }
]
'''

class LiberoVLA0(Dataset):
    def __init__(self, 
                 data_path: pathlib.Path = LIBERO_RAW_DATA_PATH,
                 horizon: int = 8,  # H: 预测未来多少个时间步
                 action_dim: int = 7,  # D: 每个动作的维度
                 num_bins: int = 1000,  # B: 离散化的bins数量
                 max_episodes: int = None):  # 最多加载多少个episode
        """
        Args:
            data_path: libero数据集根目录
            horizon: 预测的时间步数H
            action_dim: 动作维度D
            num_bins: 离散化bins数量B
            max_episodes: 限制加载的episode数量，用于调试
        """
        super().__init__()
        self.data_path = data_path
        self.horizon = horizon
        self.action_dim = action_dim
        self.num_bins = num_bins
        
        # 收集所有episode数据
        self.episodes = []
        episode_dirs = sorted(list(data_path.glob('*/')))
        
        if max_episodes:
            episode_dirs = episode_dirs[:max_episodes]
            
        for episode_dir in episode_dirs:
            main_image_folder = episode_dir / 'images'
            gripper_image_folder = episode_dir / 'gripper_images'
            action_folder = episode_dir / 'actions'
            instruction_file = episode_dir / 'instruction.txt'
            
            if not all([main_image_folder.exists(), 
                       gripper_image_folder.exists(),
                       action_folder.exists(), 
                       instruction_file.exists()]):
                continue
                
            main_images = sorted(list(main_image_folder.glob('*.jpg')))
            gripper_images = sorted(list(gripper_image_folder.glob('*.jpg')))
            actions = sorted(list(action_folder.glob('*.npy')))
            
            if len(main_images) == 0 or len(actions) == 0:
                continue
                
            instruction = instruction_file.read_text().strip()
            
            self.episodes.append({
                'episode_dir': episode_dir,
                'main_images': main_images,
                'gripper_images': gripper_images,
                'actions': actions,
                'instruction': instruction
            })
        
        # 为每个episode的每个时间步创建样本索引
        # 每个样本是 (episode_idx, timestep_idx)
        self.samples = []
        for ep_idx, episode in enumerate(self.episodes):
            # 对于每个时间步t，我们需要预测未来horizon个动作
            # 因此有效的起始时间步是 [0, len(actions) - horizon]
            num_timesteps = len(episode['actions'])
            for t in range(max(0, num_timesteps - horizon + 1)):
                self.samples.append((ep_idx, t))
        
        # 计算动作的归一化统计信息（min/max用于离散化）
        self._compute_action_stats()
    
    def _compute_action_stats(self):
        """计算所有动作的最小值和最大值，用于归一化到[0, num_bins-1]"""
        all_actions = []
        for episode in self.episodes:
            for action_file in episode['actions']:
                action = np.load(action_file)
                all_actions.append(action)
        
        if len(all_actions) == 0:
            self.action_min = np.zeros(self.action_dim)
            self.action_max = np.ones(self.action_dim)
        else:
            all_actions = np.array(all_actions)
            self.action_min = all_actions.min(axis=0)
            self.action_max = all_actions.max(axis=0)
            
            # 避免除以零
            self.action_range = self.action_max - self.action_min
            self.action_range[self.action_range < 1e-6] = 1.0
    
    def _normalize_action(self, action: np.ndarray) -> List[int]:
        """将连续动作归一化到[0, num_bins-1]的整数"""
        # 归一化到[0, 1]
        normalized = (action - self.action_min) / self.action_range
        # 缩放到[0, num_bins-1]并取整
        discretized = np.clip(normalized * (self.num_bins - 1), 0, self.num_bins - 1)
        return discretized.astype(int).tolist()
    
    def save_normalization_stats(self, save_path: str):
        """
        保存归一化统计信息到JSON文件
        
        Args:
            save_path: 保存路径（.json文件）
        """
        stats = {
            'action_min': self.action_min.tolist(),
            'action_max': self.action_max.tolist(),
            'action_range': self.action_range.tolist(),
            'action_dim': self.action_dim,
            'num_bins': self.num_bins,
            'horizon': self.horizon
        }
        
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"归一化统计信息已保存到: {save_path}")
        
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        返回格式化的消息数据，符合LLM微调格式
        """
        ep_idx, t = self.samples[idx]
        episode = self.episodes[ep_idx]
        
        # 获取当前时刻的图像
        main_image_path = episode['main_images'][t] if t < len(episode['main_images']) else episode['main_images'][-1]
        gripper_image_path = episode['gripper_images'][t] if t < len(episode['gripper_images']) else episode['gripper_images'][-1]
        
        # 获取未来horizon个动作
        future_actions = []
        for h in range(self.horizon):
            action_idx = min(t + h, len(episode['actions']) - 1)
            action = np.load(episode['actions'][action_idx])
            discretized_action = self._normalize_action(action)
            future_actions.extend(discretized_action)
        
        # 将动作转换为空格分隔的字符串
        action_str = ' '.join(map(str, future_actions))
        
        # 格式化系统提示
        system_prompt = SYSTEM_PROMPT.format(
            H=self.horizon,
            D=self.action_dim,
            B=self.num_bins - 1
        )
        
        # 构建消息格式
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(main_image_path)
                    },
                    {
                        "type": "image",
                        "image": str(gripper_image_path)
                    },
                    {
                        "type": "text",
                        "text": episode['instruction']
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": action_str
                    }
                ]
            }
        ]
        
        return {
            'messages': messages,
            'episode_idx': ep_idx,
            'timestep': t
        }





class ActionNormalizer:
    """
    用于推理时的动作反归一化工具类
    可以从保存的统计信息文件中加载，无需重新创建完整的Dataset
    """
    def __init__(self, stats_path: str = None, stats_dict: Dict[str, Any] = None):
        """
        Args:
            stats_path: 统计信息JSON文件路径
            stats_dict: 或者直接传入统计信息字典
        """
        if stats_path is not None:
            stats = ActionNormalizer.load_normalization_stats(stats_path)
        elif stats_dict is not None:
            stats = stats_dict
        else:
            raise ValueError("必须提供 stats_path 或 stats_dict 之一")
        
        self.action_min = stats['action_min']
        self.action_max = stats['action_max']
        self.action_range = stats['action_range']
        self.action_dim = stats['action_dim']
        self.num_bins = stats['num_bins']
        self.horizon = stats['horizon']
    
    @staticmethod
    def load_normalization_stats(load_path: str) -> Dict[str, Any]:
        """
        从JSON文件加载归一化统计信息
        
        Args:
            load_path: 统计信息文件路径
            
        Returns:
            包含统计信息的字典
        """
        with open(load_path, 'r') as f:
            stats = json.load(f)
        
        # 转换回numpy数组
        stats['action_min'] = np.array(stats['action_min'])
        stats['action_max'] = np.array(stats['action_max'])
        stats['action_range'] = np.array(stats['action_range'])
        
        return stats

    def unnormalize(self, discretized_action: np.ndarray) -> np.ndarray:
        """
        将离散化的动作转换回连续动作值
        
        Args:
            discretized_action: 离散动作，值范围[0, num_bins-1]
                              可以是shape (action_dim,) 或 (horizon, action_dim) 或 (horizon*action_dim,)
        
        Returns:
            连续动作值，shape与输入相同
        """
        discretized_action = np.array(discretized_action)
        original_shape = discretized_action.shape
        
        # 转换到[0, 1]
        normalized = discretized_action / (self.num_bins - 1)
        # 反归一化到原始范围
        continuous_action = normalized * self.action_range + self.action_min
        
        return continuous_action.reshape(original_shape)
    
    def parse_prediction_string(self, pred_str: str) -> np.ndarray:
        """
        将模型预测的字符串解析为连续动作数组
        
        Args:
            pred_str: 空格分隔的整数字符串，例如 "100 200 150 ... "
        
        Returns:
            连续动作数组，shape为 (horizon, action_dim)
        """
        # 解析字符串为整数列表
        tokens = pred_str.strip().split()
        discretized = np.array([int(t) for t in tokens])
        
        # 检查长度
        expected_len = self.horizon * self.action_dim
        if len(discretized) != expected_len:
            raise ValueError(f"预测长度不匹配: 期望 {expected_len}, 实际 {len(discretized)}")
        
        # reshape 为 (horizon, action_dim)
        discretized = discretized.reshape(self.horizon, self.action_dim)
        
        # 反归一化
        continuous = self.unnormalize(discretized)
        
        return continuous




