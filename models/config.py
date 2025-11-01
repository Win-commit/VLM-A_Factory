import torch
Qwen2_5_VL_3B_VLA_CONFIG = {
    "pretrained_model_name_or_path":"pretrain/Qwen2.5-VL-3B-Instruct",
    "torch_dtype":torch.bfloat16,
    "attn_implementation":"flash_attention_2",
    "device_map":"auto",
    "local_files_only":True,
}