"""
快速验证脚本：测试MC Dropout不确定性的区分度

使用方法：
python validate_mc_dropout.py --model_path ./checkpoints/xxx/checkpoint.pth

功能：
1. 加载已训练的模型
2. 在测试集上计算不确定性
3. 分析不确定性是否能区分正常/异常
4. 给出是否值得使用MC Dropout的建议
"""

import argparse
import torch
import numpy as np
from exp.exp_timesnet_ad_bayesian import Exp_TimesNet_AD_Bayesian
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description='Validate MC Dropout Uncertainty')

# 基础配置
parser.add_argument('--model_path', type=str, required=True, help='path to trained model')
parser.add_argument('--data', type=str, default='PSM', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/PSM')
parser.add_argument('--mc_samples', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=32)

# 模型配置（需要与训练时一致）
parser.add_argument('--seq_len', type=int, default=100)
parser.add_argument('--enc_in', type=int, default=25)
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--num_kernels', type=int, default=6)
parser.add_argument('--e_layers', type=int, default=2)
parser.add_argument('--d_ff', type=int, default=64)

args = parser.parse_args()

# 补充必要参数
args.task_name = 'anomaly_detection'
args.model = 'TimesNet_AD_Bayesian'
args.features = 'M'
args.target = 'OT'
args.freq = 'h'
args.label_len = 48
args.pred_len = 0
args.c_out = args.enc_in
args.dec_in = args.enc_in
args.n_heads = 4
args.d_layers = 1
args.embed = 'timeF'
args.activation = 'gelu'
args.output_attention = False
args.num_workers = 0
args.use_gpu = torch.cuda.is_available()
args.gpu = 0
args.gpu_type = 'cuda'
args.use_multi_gpu = False
args.data_path = 'train.csv'

print("=" * 60)
print("MC Dropout Uncertainty Validation")
print("=" * 60)
print(f"Model: {args.model_path}")
print(f"Dataset: {args.data}")
print(f"MC Samples: {args.mc_samples}")
print("=" * 60)

# 创建实验对象
exp = Exp_TimesNet_AD_Bayesian(args)

# 加载模型
print("\n[Step 1/3] Loading model...")
if not os.path.exists(args.model_path):
    print(f"Error: Model not found at {args.model_path}")
    exit(1)

exp.model.load_state_dict(torch.load(args.model_path))
print("Model loaded successfully!")

# 获取测试数据
print("\n[Step 2/3] Loading test data...")
test_data, test_loader = exp._get_data(flag='test')
print(f"Test data: {len(test_data)} samples")

# 计算不确定性
print("\n[Step 3/3] Computing uncertainties...")
test_uncertainties = []
test_labels = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.float().to(exp.device)
        _, uncertainty = exp._mc_dropout_inference(batch_x, args.mc_samples)
        test_uncertainties.append(uncertainty.detach().cpu().numpy())
        test_labels.append(batch_y.numpy())

test_uncertainties = np.concatenate(test_uncertainties, axis=0).reshape(-1)
test_labels = np.concatenate(test_labels, axis=0).reshape(-1)

# 分析结果
print("\n" + "=" * 60)
print("Uncertainty Analysis Results")
print("=" * 60)

normal_unc = test_uncertainties[test_labels == 0]
anomaly_unc = test_uncertainties[test_labels == 1]

print(f"\nNormal points: {len(normal_unc)}")
print(f"  mean={normal_unc.mean():.6f}, std={normal_unc.std():.6f}")
print(f"\nAnomaly points: {len(anomaly_unc)}")
print(f"  mean={anomaly_unc.mean():.6f}, std={anomaly_unc.std():.6f}")

# 计算AUC
try:
    auc = roc_auc_score(test_labels, test_uncertainties)
    print(f"\nUncertainty AUC: {auc:.4f}")

    print("\n" + "=" * 60)
    print("Recommendation:")
    print("=" * 60)

    if auc > 0.65:
        print("✅ EXCELLENT! Uncertainty has strong discriminative power.")
        print("   Recommendation: Use MC Dropout with confidence.")
        print(f"   Expected improvement: +2~5% F1")
    elif auc > 0.6:
        print("✓ GOOD! Uncertainty has moderate discriminative power.")
        print("   Recommendation: Use MC Dropout, tune uncertainty_weight.")
        print(f"   Expected improvement: +1~3% F1")
    elif auc > 0.55:
        print("⚠ MARGINAL. Uncertainty has weak discriminative power.")
        print("   Recommendation: Try increasing dropout rate or mc_samples.")
        print(f"   Expected improvement: +0.5~1% F1")
    else:
        print("✗ POOR. Uncertainty cannot distinguish anomalies.")
        print("   Recommendation: MC Dropout may not help on this dataset.")
        print("   Consider other methods (frequency attention, etc.)")

    print("=" * 60)

except Exception as e:
    print(f"\nError computing AUC: {e}")

print("\nValidation complete!")
