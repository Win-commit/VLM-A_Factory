"""
Preprocessed version of LiberoVLA0 dataset
Load all data into memory or save as preprocessed files at once to avoid I/O bottlenecks during training
"""

import pathlib
import numpy as np
import json
import pickle
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


LIBERO_RAW_DATA_PATH = pathlib.Path('/liujinxin/zhy/ICLR2026/datasets/libero/data/libero_all')

SYSTEM_PROMPT = "Analyze the input image and predict robot actions for the next {H} timesteps. Each action has {D} dimensions. Output a single sequence of {H}*{D} integers (0-{B} each), representing the {H} timesteps sequentially. Provide only space-separated numbers. Nothing else."


class LiberoVLA0Preprocessed(Dataset):
    
    def __init__(self,
                 data_path: pathlib.Path = LIBERO_RAW_DATA_PATH,
                 horizon: int = 8,
                 action_dim: int = 7,
                 num_bins: int = 1000,
                 max_episodes: int = None,
                 preprocess_cache_path: Optional[str] = None):
        """
        Args:
            data_path: libero数据集根目录
            horizon: 预测的时间步数H
            action_dim: 动作维度D
            num_bins: 离散化bins数量B
            max_episodes: 限制加载的episode数量
            preprocess_cache_path: 预处理缓存文件路径（.pkl
        """
        super().__init__()
        self.data_path = data_path
        self.horizon = horizon
        self.action_dim = action_dim
        self.num_bins = num_bins
        
        if preprocess_cache_path and pathlib.Path(preprocess_cache_path).exists():
            self._load_from_cache(preprocess_cache_path)
        else:
            self._preprocess_data(data_path, max_episodes)
            
            if preprocess_cache_path:
                self._save_to_cache(preprocess_cache_path)
    
    def _preprocess_data(self, data_path: pathlib.Path, max_episodes: Optional[int]):
        """预处理所有数据"""
        episodes_meta = []
        episode_dirs = sorted(list(data_path.glob('*/')))
        
        if max_episodes:
            episode_dirs = episode_dirs[:max_episodes]
        
        for episode_dir in tqdm(episode_dirs, desc="scan episodes"):
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
            
            episodes_meta.append({
                'main_images': main_images,
                'gripper_images': gripper_images,
                'actions': actions,
                'instruction': instruction
            })
        
        all_actions = []
        for episode in tqdm(episodes_meta, desc="Load actions"):
            for action_file in episode['actions']:
                action = np.load(action_file)
                all_actions.append(action)
        
        all_actions = np.array(all_actions)
        self.action_min = all_actions.min(axis=0)
        self.action_max = all_actions.max(axis=0)
        
        self.action_range = self.action_max - self.action_min
        
        self.samples = []
        
        for ep_idx, episode in enumerate(tqdm(episodes_meta, desc="pre-process samples")):
            num_timesteps = len(episode['actions'])
            
            episode_actions = []
            for action_file in episode['actions']:
                action = np.load(action_file)
                episode_actions.append(action)
            episode_actions = np.array(episode_actions)
            
            # 为每个有效时间步创建样本
            for t in range(max(0, num_timesteps - self.horizon + 1)):
                main_img = str(episode['main_images'][min(t, len(episode['main_images'])-1)])
                gripper_img = str(episode['gripper_images'][min(t, len(episode['gripper_images'])-1)])
                
                future_actions = []
                for h in range(self.horizon):
                    action_idx = min(t + h, len(episode_actions) - 1)
                    action = episode_actions[action_idx]
                    discretized = self._normalize_action(action)
                    future_actions.extend(discretized)
                
                # 转换为字符串
                action_str = ' '.join(map(str, future_actions))
                
                # 构建消息
                sample = {
                    'main_image_path': main_img,
                    'gripper_image_path': gripper_img,
                    'instruction': episode['instruction'],
                    'action_str': action_str,
                    'episode_idx': ep_idx,
                    'timestep': t
                }
                
                self.samples.append(sample)
    
    def _normalize_action(self, action: np.ndarray) -> List[int]:
        normalized = (action - self.action_min) / self.action_range
        discretized = np.clip(normalized * (self.num_bins - 1), 0, self.num_bins - 1)
        return discretized.astype(int).tolist()
    
    def _save_to_cache(self, cache_path: str):
        cache_path = pathlib.Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        cache_data = {
            'samples': self.samples,
            'action_min': self.action_min,
            'action_max': self.action_max,
            'action_range': self.action_range,
            'horizon': self.horizon,
            'action_dim': self.action_dim,
            'num_bins': self.num_bins,
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # 同时保存归一化统计信息为JSON（用于推理）
        stats_path = cache_path.parent / 'action_norm_stats.json'
        self.save_normalization_stats(str(stats_path))
    
    def _load_from_cache(self, cache_path: str):
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.samples = cache_data['samples']
        self.action_min = cache_data['action_min']
        self.action_max = cache_data['action_max']
        self.action_range = cache_data['action_range']
        self.horizon = cache_data['horizon']
        self.action_dim = cache_data['action_dim']
        self.num_bins = cache_data['num_bins']
    
    def save_normalization_stats(self, save_path: str):
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
        
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # 格式化系统提示
        system_prompt = SYSTEM_PROMPT.format(
            H=self.horizon,
            D=self.action_dim,
            B=self.num_bins - 1
        )
        
        # 构建消息格式（已经预处理好）
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
                        "image": f"file://{sample['main_image_path']}"
                    },
                    {
                        "type": "image",
                        "image": f"file://{sample['gripper_image_path']}"
                    },
                    {
                        "type": "text",
                        "text": sample['instruction']
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": sample['action_str']
                    }
                ]
            }
        ]
        

        return messages


class ActionNormalizer:
    """用于推理时的动作反归一化工具类"""
    
    def __init__(self, stats_path: str = None, stats_dict: Dict[str, Any] = None):
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
        with open(load_path, 'r') as f:
            stats = json.load(f)
        
        stats['action_min'] = np.array(stats['action_min'])
        stats['action_max'] = np.array(stats['action_max'])
        stats['action_range'] = np.array(stats['action_range'])
        
        return stats
    
    def unnormalize(self, discretized_action: np.ndarray) -> np.ndarray:
        discretized_action = np.array(discretized_action)
        original_shape = discretized_action.shape
        
        normalized = discretized_action / (self.num_bins - 1)
        continuous_action = normalized * self.action_range + self.action_min
        
        return continuous_action.reshape(original_shape)
    
    def parse_prediction_string(self, pred_str: str) -> np.ndarray:
        tokens = pred_str.strip().split()
        discretized = np.array([int(t) for t in tokens])
        
        assert len(discretized) == self.horizon * self.action_dim, f"预测长度不匹配: 期望 {self.horizon * self.action_dim}, 实际 {len(discretized)}"
       
        discretized = discretized.reshape(self.horizon, self.action_dim)
        return self.unnormalize(discretized)


if __name__ == "__main__":
    print("创建预处理数据集...")
    
    # 第一次运行：预处理并保存缓存
    dataset = LiberoVLA0Preprocessed(
        horizon=8,
        action_dim=7,
        num_bins=1000,
        preprocess_cache_path="/liujinxin/zhy/lirunze/vla-0/dataset/libero/libero_preprocessed_8horizon_1000bins.pkl"
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    print(f"第一个样本: {dataset[0][2]['content'][0]['text'][:50]}...")
    
    # # 后续运行
    # dataset2 = LiberoVLA0Preprocessed(
    #     preprocess_cache_path="./cache/libero_preprocessed.pkl",
    # )
    # print(f"数据集大小: {len(dataset2)}")

