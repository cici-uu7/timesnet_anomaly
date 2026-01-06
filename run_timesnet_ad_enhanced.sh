#!/bin/bash

# TimesNet_AD 增强版模型运行脚本
# 所有关键参数都可以在此配置
# 用法: ./run_timesnet_ad_enhanced.sh

echo "==========================================="
echo "TimesNet_AD 增强版模型"
echo "特性: 多层架构 + 动态Prior + Minimax训练"
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
  --e_layers 2 \
  --top_k 3 \
  --num_kernels 6 \
  --enc_in 25 \
  --c_out 25 \
  --n_heads 8 \
  --anomaly_ratio 1.0 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --train_epochs 5 \
  --patience 3 \
  --k 0.5 \
  --margin 0.5 \
  --alpha 0.6 \
  --beta 0.4 \
  --sigma_init_factor 5.0 \
  --dynamic_prior True \
  --fusion_method weighted \
  --output_attention

echo ""
echo "==========================================="
echo "训练完成!"
echo "==========================================="

