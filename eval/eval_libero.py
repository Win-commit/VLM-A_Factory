import os
# os.environ["MUJOCO_GL"] = "egl"
import torch
import pickle
import numpy as np
from time import time
import sys
sys.path.append("/liujinxin/zhy/lirunze/vla-0/models")
from Qwen2_5_VL_3B_VLA import Qwen2_5_VL_3B_VLA
from PIL import Image
from torch.nn.functional import cross_entropy
from random import shuffle
import random
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union, List
from collections import deque
import tqdm
import argparse
from libero_utility import *


class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"

TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 520,
    TaskSuite.LIBERO_90: 400,
}

# 每次预测的动作步数
NUM_ACTIONS_CHUNK = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@dataclass
class EvaluationConfig:
    # 模型相关参数
    pretrained_checkpoint: str = ""
    env_ckpt: Optional[str] = None
    vision_hub: str = "/liujinxin/zhy/ICLR2026/pretrain/Emu3-VisionTokenizer"
    fast_path: str = "/liujinxin/zhy/ICLR2026/pretrain/fast"
    parallel_mode: bool = True
    parallel_reward_groups: int = 10
    reward_group_size: int = 10
    num_open_loop_steps: int = 10
    visual_token_pattern: str = "<|visual token {token_id:0>6d}|>"
    noise_factor: float = 0.4
    gamma: float = 0.9
    
    # LIBERO环境参数
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256
    window_size: int = 1 #0就退化成了不加任何历史
    
    # 工具参数
    run_id_note: Optional[str] = None
    local_log_dir: str = "/liujinxin/zhy/ICLR2026/eval/logs"
    save_videos: bool = True
    use_wandb: bool = False
    wandb_entity: str = ""
    wandb_project: str = ""
    seed: int = 7