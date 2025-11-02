import torch
import random
import copy
import string
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import Dict, Optional, Union, List



class Qwen2_5_VL_3B_VLA(Qwen2_5_VLForConditionalGeneration):
    '''
    Wrap Qwen2.5-VL-3B-Instruct for VLA mode.
    '''
    pass
