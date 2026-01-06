#!/bin/bash

# TimesNet_AD_V2 运行脚本
# 最后层融合架构 (Last-Layer Fusion)
# 用法: ./run_timesnet_ad_v2.sh

echo "==========================================="
echo "TimesNet_AD_V2 - 最后层融合架构"
echo "==========================================="
echo "设计理念:"
echo "  1. 前 N 层: 纯 TimesBlock (保持重构能力)"
echo "  2. 最后层: AnomalyAttention (关联差异检测)"
echo "  3. 异常分数 = α * 重构误差 + β * 关联差异"
echo "==========================================="
echo ""
echo "核心优势:"
echo "  - TimesNet 频域特征不受干扰"
echo "  - AnomalyAttention 利用高层语义特征"
echo "  - 两种信号独立计算，最后加权组合"
echo "==========================================="

python run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/PSM \
  --model_id PSM \
  --model TimesNet_AD_V2 \
  --data PSM \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 3 \
  --top_k 3 \
  --num_kernels 6 \
  --enc_in 25 \
  --c_out 25 \
  --n_heads 8 \
  --anomaly_ratio 1.0 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 5 \
  --lradj type3 \
  --k 1.0 \
  --alpha 0.5 \
  --beta 0.5 \
  --dropout 0.1 \
  --output_attention

echo ""
echo "==========================================="
echo "训练完成!"
echo "==========================================="
echo ""
echo "调参建议:"
echo "  - 若重构能力不足: 增加 alpha, 减少 beta"
echo "  - 若关联差异无效: 增加 beta, 减少 k"
echo "  - 若 Vali Loss 过高: 减少 k (保护重构)"
echo "==========================================="
