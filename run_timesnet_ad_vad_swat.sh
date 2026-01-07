#!/bin/bash

echo "==========================================="
echo "TimesNet_AD_VAD - SWaT Dataset"
echo "==========================================="
echo ""
echo "Dataset: SWaT (Secure Water Treatment)"
echo "  Variables: 51 (sensors + actuators)"
echo "  Expected: Strong physical coupling (water treatment process)"
echo "  VAD适配度: ⭐⭐⭐⭐⭐ (工业控制系统,物理约束极强)"
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

# SWaT 数据集参数
# - 51 个变量（传感器+执行器）
# - 水处理系统，物理耦合强
# - 适合 VAD（变量间有明确的物理关系）

python run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SWaT \
  --model_id SWAT \
  --model TimesNet_AD_VAD \
  --data SWAT \
  --features M \
  --seq_len 100 \
  --label_len 48 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 2 \
  --top_k 3 \
  --num_kernels 6 \
  --enc_in 51 \
  --c_out 51 \
  --n_heads 4 \
  --anomaly_ratio 1.0 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 5 \
  --lradj type3 \
  --beta 0.5 \
  --prior_lambda 0.0 \
  --dropout 0.1

echo ""
echo "==========================================="
echo "训练完成!"
echo "==========================================="
echo ""
echo "参数说明:"
echo "  --beta: VAD boost 权重 (默认0.5)"
echo "  --prior_lambda: Prior融合权重 (默认0.0)"
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
echo "  - β建议: 0.5~1.0 (SWaT物理耦合强,可以用更大β)"
echo ""
echo "SWaT 特点:"
echo "  - 水处理系统,传感器间有明确的因果关系"
echo "  - 例如: 水泵状态 → 流量 → 水位 → 压力"
echo "  - VAD 可以捕捉这些物理关系的破坏"
echo "  - 异常通常是单点故障(传感器/执行器失效)"
echo "  - 预期 VAD 效果好于 SMAP（SWaT是局部异常,SMAP可能是系统性异常）"
echo "==========================================="
