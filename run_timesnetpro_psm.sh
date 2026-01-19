#!/bin/bash

# TimesNetPro 运行脚本
# TimesNet with Adaptive Period Attention
# 自适应周期注意力机制改进的 TimesNet

echo "==========================================="
echo "TimesNetPro - Adaptive Period Attention"
echo "==========================================="
echo ""
echo "核心改进："
echo "  1. 自适应周期注意力机制"
echo "  2. 可学习的周期权重融合"
echo "  3. 更好地捕捉周期异常"
echo ""
echo "技术细节："
echo "  - 结合 FFT 权重（频域先验）和学习权重（数据驱动）"
echo "  - 使用 Query-Key 注意力机制动态判断周期重要性"
echo "  - 保持 TimesNet 核心架构（2D Conv）不变"
echo "==========================================="
echo ""

python run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/PSM \
  --model_id PSM \
  --model TimesNetPro \
  --data PSM \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 2 \
  --top_k 5 \
  --num_kernels 6 \
  --enc_in 25 \
  --c_out 25 \
  --n_heads 4 \
  --anomaly_ratio 1.0 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 5 \
  --lradj type3 \
  --dropout 0.1

echo ""
echo "==========================================="
echo "训练完成!"
echo "==========================================="
echo ""
echo "模型说明："
echo "  TimesNetPro 在 TimesNet 基础上添加了自适应周期注意力机制"
echo "  能够更好地识别哪些周期对异常检测更重要"
echo ""
echo "参数说明："
echo "  --top_k: Top-k 周期数量（默认5）"
echo "  --d_model: 模型维度（默认64）"
echo "  --dropout: Dropout 比率（默认0.1）"
echo ""
echo "预期改进："
echo "  - 更好的周期异常检测能力"
echo "  - 保持 TimesNet 的重构能力"
echo "  - 自适应学习周期重要性"
echo "==========================================="
