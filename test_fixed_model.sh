#!/bin/bash

# 测试修复后的TimesNet_AD模型
# 使用PSM数据集进行异常检测

echo "========================================="
echo "测试改进后的TimesNet_AD模型"
echo "========================================="

# 可调参数配置
K_VAL=3.0                    # Minimax策略权重 (推荐: 1.0-5.0)
MARGIN=0.5                   # Prior差异阈值 (推荐: 0.1-1.0)
ALPHA=0.7                    # 重构误差权重 (推荐: 0.3-0.7)
BETA=0.3                     # 关联差异权重 (推荐: 0.3-0.7)
SIGMA_INIT_FACTOR=5.0        # Prior sigma初始化因子 (推荐: 3.0-10.0)

echo "参数配置:"
echo "  k (Minimax权重):         $K_VAL"
echo "  margin (差异阈值):       $MARGIN"
echo "  alpha (重构权重):        $ALPHA"
echo "  beta (关联差异权重):     $BETA"
echo "  sigma_init_factor:       $SIGMA_INIT_FACTOR"
echo "========================================="

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
  --batch_size 128 \
  --learning_rate 0.0001 \
  --train_epochs 5 \
  --patience 3 \
  --k $K_VAL \
  --margin $MARGIN \
  --alpha $ALPHA \
  --beta $BETA \
  --sigma_init_factor $SIGMA_INIT_FACTOR \
  --anomaly_ratio 1.0 \
  --output_attention

echo ""
echo "========================================="
echo "训练和测试完成!"
echo "========================================="
echo ""
echo "调参建议:"
echo "1. 如果模型过拟合异常 → 降低 k (如 1.5-2.0)"
echo "2. 如果检测率太低 → 提高 beta, 降低 alpha"
echo "3. 如果误报率太高 → 提高 alpha, 降低 beta"
echo "4. 如果loss震荡 → 降低 margin (如 0.1-0.3)"
echo "5. 如果Prior学习不稳定 → 调整 sigma_init_factor (3-10)"
echo "========================================="
