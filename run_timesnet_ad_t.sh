#!/bin/bash

# TimesNet_AD_T 运行脚本
# 双向交互融合架构 (Bidirectional Interaction Fusion)
# 用法: ./run_timesnet_ad_t.sh

echo "==========================================="
echo "TimesNet_AD_T - 双向交互融合架构"
echo "==========================================="
echo "创新点:"
echo "  1. 层内融合: TimesBlock + AnomalyAttention 在同一层"
echo "  2. 双向交互: 频域↔关联 互相增强"
echo "  3. 周期感知Prior: FFT信息指导Prior生成"
echo "==========================================="

python run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/PSM \
  --model_id PSM \
  --model TimesNet_AD_T \
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
  --train_epochs 5 \
  --patience 3 \
  --k 3.0 \
  --dropout 0.1 \
  --output_attention

echo ""
echo "==========================================="
echo "训练完成!"
echo "==========================================="
echo ""
echo "模型特性:"
echo "  - TimesBlock_AD: 层内融合 FFT + Anomaly Attention"
echo "  - BidirectionalInteraction: 频域→关联 + 关联→频域"
echo "  - 可学习融合权重: alpha, beta"
echo ""
echo "预期效果:"
echo "  - 重构误差: 接近原版 TimesNet (~0.01)"
echo "  - F1 Score: 97.5%+ (超越两个基线)"
echo "==========================================="
