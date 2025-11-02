import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wandb
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    AutoProcessor,
    HfArgumentParser,
    Qwen2_5_VLForConditionalGeneration
)
from dataclasses import dataclass, field
from typing import Optional
import pathlib

from dataset.libero_dataset_preprocessed import LiberoVLA0Preprocessed
from dataset.InputTransformers import InputTransformers

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="/liujinxin/zhy/lirunze/vla-0/pretrain/Qwen2.5-VL-3B-Instruct")
    attn_type: str = field(default="flash_attention_2")
    local_files_only: bool = field(default=True)
    trust_remote_code: bool = field(default=True)


@dataclass
class DataArguments:
    dataset_cache_path: str = field(default=None)
    action_mask_ratio: float = field(default=0.3)
    max_samples: Optional[int] = field(default=None)
    horizon: int = field(default=8)
    action_dim: int = field(default=7)
    num_bins: int = field(default=1000)



@dataclass
class TrainingArguments(TrainingArguments):
    exp_name: str = field(default="")
    output_dir: str = field(default=None)
    num_train_epochs: int = field(default=64)
    per_device_train_batch_size: int = field(default=24)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=5e-6)
    warmup_steps: int = field(default=0)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500)
    bf16: bool = field(default=True)
    dataloader_pin_memory: bool = field(default=True)
    dataloader_num_workers: int = field(default=12)
    report_to: str = field(default="wandb")
    eval_strategy: str = field(default = "no") 
    min_learning_rate: Optional[float] = field(default=None)
    
    def __post_init__(self):
        super().__post_init__()
        # Set up lr_scheduler_kwargs if min_learning_rate is provided
        if self.min_learning_rate is not None:
            if not hasattr(self, 'lr_scheduler_kwargs') or self.lr_scheduler_kwargs is None:
                self.lr_scheduler_kwargs = {}
            self.lr_scheduler_kwargs["min_lr"] = self.min_learning_rate
    


class VLADataCollator:
    
    def __init__(self, processor: AutoProcessor, action_mask_ratio: float = 0.0):
        self.processor = processor
        self.action_mask_ratio = action_mask_ratio
    
    def __call__(self, batch):
        inputs = InputTransformers.transform_input(
            processor=self.processor,
            messages=batch,
            action_mask_ratio=self.action_mask_ratio
        )
        
        return inputs


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir,exist_ok=True)
        
    # Initialize wandb
    if training_args.report_to and "wandb" in training_args.report_to:
        if training_args.resume_from_checkpoint:
            run_id = (pathlib.Path(training_args.output_dir).resolve() / "wandb_id.txt").read_text().strip()
            wandb.init(
                project=training_args.exp_name,
                id=run_id, 
                resume="must"
            )
        else:
            wandb.init(
                project=training_args.exp_name,
                name=f"training_{training_args.run_name or 'default'}",
                config={
                    "model_name_or_path": model_args.model_name_or_path,
                    "action_mask_ratio": data_args.action_mask_ratio,
                    "horizon": data_args.horizon,
                    "action_dim": data_args.action_dim,
                    "num_bins": data_args.num_bins,
                    "num_train_epochs": training_args.num_train_epochs,
                    "per_device_train_batch_size": training_args.per_device_train_batch_size,
                    "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                    "learning_rate": training_args.learning_rate,
                    "warmup_steps": training_args.warmup_steps,
                    "logging_steps": training_args.logging_steps,
                    "bf16": training_args.bf16,
                    "dataloader_pin_memory": training_args.dataloader_pin_memory,
                    "dataloader_num_workers": training_args.dataloader_num_workers,
                }
            )
            (pathlib.Path(training_args.output_dir).resolve() / "wandb_id.txt").write_text(wandb.run.id)

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code
    )
    
    processor.tokenizer.padding_side = "right" 
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_type,
        local_files_only=model_args.local_files_only
    )
    
    if hasattr(training_args, "gradient_checkpointing") and training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    train_dataset = LiberoVLA0Preprocessed(
        preprocess_cache_path=data_args.dataset_cache_path,
        horizon=data_args.horizon,
        action_dim=data_args.action_dim,
        num_bins=data_args.num_bins
    )
    
    
    # 创建数据收集器
    data_collator = VLADataCollator(
        processor=processor,
        action_mask_ratio=data_args.action_mask_ratio
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    if training_args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()
    
    trainer.save_model()
    trainer.save_state()
    processor.save_pretrained(training_args.output_dir)
    


if __name__ == "__main__":
    main()
