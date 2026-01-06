#!/bin/bash

# TimesNet_AD 原版模型运行脚本
# 用法: ./run_timesnet_ad.sh [train|test]

MODE=${1:-train}  # 默认训练模式

echo "==========================================="
echo "TimesNet_AD 原版模型"
echo "==========================================="

python run.py \
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
  --anomaly_ratio 1.0 \
  --batch_size 128 \
  --train_epochs 5 \
  --k 3.0 \
  --margin 0.5 \
  --alpha 0.6 \
  --beta 0.4 \
  --sigma_init_factor 5.0 \
  --output_attention

echo ""
echo "==========================================="
echo "完成!"
echo "==========================================="
