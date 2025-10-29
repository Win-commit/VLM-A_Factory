import torch
from transformers import Qwen2_5_VLForConditionalGeneration
from typing import Dict,Optional,Union



class Qwen2_5_VL_3B_VLA(Qwen2_5_VLForConditionalGeneration):
    def __init__(self):
        super().__init__()

    def transform_input(self, message:str, action_mask_ratio:float = 0.0):
        '''
        将string的message输入转为token，以及配套的atten mask 和label
        并且针对vla-0实现action的随机遮掩
        '''
        '''
        test:
        text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
        )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    回头看一下这些都是什么
        '''
