#!/bin/bash

# 检查 Checkpoint 目录
if [ ! -d "./checkpoints" ]; then
    mkdir ./checkpoints
fi

# PSM 数据集参数设置
# seq_len=100, pred_len=0 (AD任务不需要预测未来)
# d_model=64, d_ff=64 平衡显存
# k=3.0 为关联差异系数

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/PSM \
  --model_id PSM \
  --model TimesNet_AD \
  --data PSM \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 3 \
  --enc_in 25 \
  --c_out 25 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 5 \
  --learning_rate 0.0001 \
  --k 3.0 \
  --itr 1