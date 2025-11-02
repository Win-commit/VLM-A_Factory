import torch
import random
import copy
import string
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import Dict, Optional, Union, List



class InputTransformers():
    
    @staticmethod
    def transform_input(processor: AutoProcessor, messages: List, action_mask_ratio: float = 0.0):
        '''
        Args:
            messages: 单个消息列表 [{role:..., content:...}, ...] 
                     或批量消息列表 [[{role:..., content:...}, ...], [...]]
            action_mask_ratio: Proportion of action partially randomized masking (0.0-1.0)
        
        Returns:
            Dict包含:
                - input_ids
                - attention_mask
                - labels Labels used to calculate losses (token ids for the assistant part, -100 for the rest)
                - other_keys
        '''
        assert processor is not None, "Processor is not set"

        is_batch = isinstance(messages[0], list)
        
        if not is_batch:
            original_messages_list = [messages]
        else:
            original_messages_list = messages
        
        original_texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in original_messages_list
        ]
        
        image_inputs, video_inputs = process_vision_info(original_messages_list)
        
        labels_inputs = processor(
            text=original_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        labels = labels_inputs['input_ids'].clone()

        if action_mask_ratio > 0.0:
            messages_masked = copy.deepcopy(messages)
            messages_masked = InputTransformers.mask_actions(messages_masked, action_mask_ratio, is_batch=is_batch)
            
            if not is_batch:
                messages_masked_list = [messages_masked]
            else:
                messages_masked_list = messages_masked
            
            masked_texts = [
                processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
                for msg in messages_masked_list
            ]
            
            inputs = processor(
                text=masked_texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

        else:
            inputs = labels_inputs
        
        for idx, original_msgs in enumerate(original_messages_list):
            # 移除assistant消息
            msgs_without_assistant = [msg for msg in original_msgs if msg['role'] != 'assistant']
            
            text_without_assistant = processor.apply_chat_template(
                msgs_without_assistant, tokenize=False, add_generation_prompt=True
            )
            
            tokens_without_assistant = processor(
                text=text_without_assistant,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            assistant_start_idx = tokens_without_assistant['attention_mask'][0].sum().item()
            
            labels[idx, :assistant_start_idx] = -100
        
        labels[inputs['attention_mask'] == 0] = -100
        
        inputs['labels'] = labels
    
        return inputs
    
    @staticmethod
    def mask_actions(messages: List, mask_ratio: float, is_batch: bool = False) -> List:
        def mask_single_message(msg_list):
            for msg in msg_list:
                if msg['role'] == 'assistant':
                    for content_item in msg['content']:
                        if content_item['type'] == 'text':
                            original_text = content_item['text']
                            if len(original_text) > 0:
                                digit_indices = [i for i, char in enumerate(original_text) if char.isdigit()]
                                
                                if len(digit_indices) > 0:
                                    num_to_mask = int(len(digit_indices) * mask_ratio)
                                    if num_to_mask > 0:
                                        mask_indices = random.sample(digit_indices, num_to_mask)
                                        
                                        chars = list(original_text)
                                        
                                        for idx in mask_indices:
                                            original_digit = int(chars[idx])
                                            other_digits = [d for d in range(10) if d != original_digit]
                                            chars[idx] = str(random.choice(other_digits))
                                        
                                        content_item['text'] = ''.join(chars)
            return msg_list
        
        if is_batch:
            return [mask_single_message(msg_list) for msg_list in messages]
        else:
            return mask_single_message(messages)