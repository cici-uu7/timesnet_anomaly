#!/bin/bash

# TimesNet_AD 增强版模型运行脚本
# 特性: 多层架构 + 多层级融合（暂时关闭动态Prior以保持稳定性）
# 用法: ./run_timesnet_ad_enhanced.sh

echo "==========================================="
echo "TimesNet_AD 增强版模型"
echo "特性: 多层架构 + 多层级融合"
echo "==========================================="

python run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/PSM \
  --model_id PSM \
  --model TimesNet_AD_Enhanced \
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
  --fusion_method weighted \
  --output_attention

echo ""
echo "==========================================="
echo "完成!"
echo "==========================================="
