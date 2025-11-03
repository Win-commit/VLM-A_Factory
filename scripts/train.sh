WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
NGPUS=8

export WANDB_BASE_URL="https://api.bandw.top"


export TORCH_EXTENSIONS_DIR=/tmp
DATAPATH='/liujinxin/zhy/lirunze/vla-0/dataset/libero/libero_preprocessed_8horizon_1000bins.pkl'

EXP_NAME="VLA-0_8horizon_1000bins_actionMask0.4_ignoreMaskToken_try1"

export PYTHONPATH=$(pwd)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export CUDA_HOME=/usr/local/cuda-11.7
# export LD_LIBRARY_PATH=/usr/local/cuda-11.7/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
# export LIBRARY_PATH=/usr/local/cuda-11.7/targets/x86_64-linux/lib:$LIBRARY_PATH

torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=1 \
    --node_rank=${RANK} \
    train/train.py \
    --model_name_or_path /liujinxin/zhy/lirunze/vla-0/pretrain/Qwen2.5-VL-3B-Instruct \
    --deepspeed /liujinxin/zhy/lirunze/vla-0/train/deepspeed/zero3.json \
    --output_dir "logs/"${EXP_NAME} \
    --learning_rate 5e-6 \
    --min_learning_rate 5e-6 \
    --max_grad_norm 5.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --bf16 True \
    --tf32 True \
    --dataset_cache_path ${DATAPATH} \
    --action_mask_ratio 0.4 \
    --horizon 8 \
    --action_dim 7 \
    --num_bins 1000 \
    --num_train_epochs 64 \
    --dataloader_num_workers 12 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --warmup_steps 0 \
    --per_device_train_batch_size 40 \
    --seed 42 \
    --logging_steps 8 \
    --attn_type "flash_attention_2" \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --save_total_limit 8 \
    --eval_strategy no \
    --exp_name $EXP_NAME \
    --report_to wandb
