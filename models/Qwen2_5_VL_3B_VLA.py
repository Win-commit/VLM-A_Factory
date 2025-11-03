import torch
import random
import copy
import string
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor,LogitsProcessor, LogitsProcessorList
from qwen_vl_utils import process_vision_info
from typing import Dict, Optional, Union, List
import numpy as np
import json
from typing import Any

class ActionConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids, scores):
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if mask.ndim == 1:
            mask[self.allowed_token_ids] = True
        else:
            mask[:, self.allowed_token_ids] = True
        scores[~mask] = -float("inf")
        return scores


class ActionUnnormalizer:
    """Inverse normalization tools for actions in inference"""
    def __init__(self, stats_path: Optional[str] = None, stats_dict: Optional[Dict[str, Any]] = None):
        if stats_path is not None:
            stats = ActionUnnormalizer.load_normalization_stats(stats_path)
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
        normalized_actions = pred_str.strip().split()
        discretized = np.array([int(t) for t in normalized_actions])
        
        assert len(discretized) == self.horizon * self.action_dim, f"预测长度不匹配: 期望 {self.horizon * self.action_dim}, 实际 {len(discretized)}"
       
        discretized = discretized.reshape(self.horizon, self.action_dim)
        return self.unnormalize(discretized)

class Qwen2_5_VL_3B_VLA:
    '''
    Wrap Qwen2.5-VL-3B-Instruct for VLA mode.
    '''
    def __init__(self, model_path: str, action_norm_stats_path: str):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path = model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            local_files_only=True,
            device_map="auto"
        )
        self.model.eval()
        #Tmp:存model的时候忘存tokenizer了，不过无所谓，先暂时用着
        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path = "/liujinxin/zhy/lirunze/vla-0/pretrain/Qwen2.5-VL-3B-Instruct",
            trust_remote_code=True
        )
        self.action_unnormalizer = ActionUnnormalizer(stats_path=action_norm_stats_path)
        self.SYSTEM_PROMPT = "Analyze the input image and predict robot actions for the next {H} timesteps. Each action has {D} dimensions. Output a single sequence of {H}*{D} integers (0-{B} each), representing the {H} timesteps sequentially. Provide only space-separated numbers. Nothing else.".format(H=self.action_unnormalizer.horizon, D=self.action_unnormalizer.action_dim, B=self.action_unnormalizer.num_bins - 1)

        # Only spaces, numbers 0-9 and EOS are allowed.
        self.allowed_token_ids = self._compute_allowed_token_ids()

    
    def _build_messages(self, task_instruction:str, main_image: str, gripper_image: str) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": self.SYSTEM_PROMPT
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image;base64,{main_image}"
                    },
                    {
                        "type": "image",
                        "image": f"data:image;base64,{gripper_image}"
                    },
                    {
                        "type": "text",
                        "text": task_instruction
                    }
                ]  
            },
        ]
        return messages
    
    def _compute_allowed_token_ids(self) -> List[int]:
        tokenizer = self.processor.tokenizer
        allowed_chars = " 0123456789"

        allowed: List[int] = []
        for ch in allowed_chars:
            token_ids = tokenizer.encode(ch, add_special_tokens=False)
            allowed.extend(token_ids)

        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None:
            eos_id = getattr(self.model.config, "eos_token_id", None)
        if eos_id is not None:
            allowed.append(int(eos_id))
        else:
            raise ValueError("EOS token ID is not set")
            

        return sorted(set(allowed))


    def generate_actions(self, task_instruction:str, main_image: str, gripper_image: str) -> np.ndarray:
        '''
        Args:
            task_instruction: str, the instruction of the task
            main_image: str, the base64 encoded main image of the task
            gripper_image: str, the base64 encoded gripper image of the task
        Returns:
            np.ndarray, the predicted actions
        '''
        messages = self._build_messages(task_instruction, main_image, gripper_image)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        logits_processor = LogitsProcessorList([
            ActionConstraintLogitsProcessor(self.allowed_token_ids)
        ])

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.action_unnormalizer.horizon * self.action_unnormalizer.action_dim * 10,
            logits_processor=logits_processor,
            do_sample=False  # Using Greedy Sampling
        )

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        predicted_actions = self.action_unnormalizer.parse_prediction_string(output_text[0])

        return predicted_actions