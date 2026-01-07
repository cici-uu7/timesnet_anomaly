#!/bin/bash

# TimesNet_AD_VAD 运行脚本
# TimesNet + Variable Association Discrepancy

echo "==========================================="
echo "TimesNet_AD_VAD - Variable Association Discrepancy"
echo "==========================================="
echo ""
echo "Prior 融合策略:"
echo "  hybrid_prior = λ * physical + (1-λ) * statistical"
echo ""
echo "  λ = 1.0: 完全信任物理约束"
echo "  λ = 0.5: 物理和统计各占一半（推荐）"
echo "  λ = 0.0: 完全用统计（无物理知识时）"
echo "==========================================="
echo ""
echo "异常分数 (VAD Boost Only):"
echo "  vad_boost = max(0, (vad - mean) / std)"
echo "  Score = rec_error * (1 + β * vad_boost)"
echo ""
echo "  特点: VAD 只会放大异常分数，不会降低"
echo "        避免 VAD 不准时影响 TimesNet 性能"
echo "==========================================="

python run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/PSM \
  --model_id PSM \
  --model TimesNet_AD_VAD \
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
  --n_heads 4 \
  --anomaly_ratio 1.0 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 5 \
  --lradj type3 \
  --beta 0.3 \
  --prior_lambda 0 \
  --dropout 0.1

echo ""
echo "==========================================="
echo "训练完成!"
echo "==========================================="
echo ""
echo "参数说明:"
echo "  --beta: VAD boost 权重 (默认0.3)"
echo "  --prior_lambda: Prior融合权重 (默认0.5)"
echo ""
echo "融合策略说明:"
echo "  VAD Boost Only: VAD只放大异常分数，不会降低"
echo "  vad_boost = max(0, (vad - mean) / std)"
echo "  score = rec * (1 + beta * vad_boost)"
echo ""
echo "调参建议:"
echo "  - 无物理知识: --prior_lambda 0.0"
echo "  - 有物理知识: --prior_lambda 0.5~0.7"
echo "  - 完全信任物理: --prior_lambda 1.0"
echo "==========================================="
