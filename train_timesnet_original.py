"""
训练原版TimesNet模型（用于后续MC Dropout测试）

使用方法：
python train_timesnet_original.py
"""

import argparse
import torch
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
import random
import numpy as np

# 固定随机种子
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Train Original TimesNet')

# 基础配置
parser.add_argument('--task_name', type=str, default='anomaly_detection')
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--model_id', type=str, default='PSM')
parser.add_argument('--model', type=str, default='TimesNet')

# 数据配置
parser.add_argument('--data', type=str, default='PSM')
parser.add_argument('--root_path', type=str, default='./dataset/PSM')
parser.add_argument('--data_path', type=str, default='train.csv')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--freq', type=str, default='h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

# 时序配置
parser.add_argument('--seq_len', type=int, default=100)
parser.add_argument('--label_len', type=int, default=48)
parser.add_argument('--pred_len', type=int, default=0)

# 模型配置
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--num_kernels', type=int, default=6)
parser.add_argument('--enc_in', type=int, default=25)
parser.add_argument('--dec_in', type=int, default=25)
parser.add_argument('--c_out', type=int, default=25)
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--e_layers', type=int, default=2)
parser.add_argument('--d_layers', type=int, default=1)
parser.add_argument('--d_ff', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.1)  # 关键！
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--activation', type=str, default='gelu')
parser.add_argument('--output_attention', action='store_true')

# 训练配置
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--itr', type=int, default=1)
parser.add_argument('--train_epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--loss', type=str, default='MSE')
parser.add_argument('--lradj', type=str, default='type1')

# GPU配置
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--gpu_type', type=str, default='cuda', choices=['cuda', 'mps'])
parser.add_argument('--use_multi_gpu', action='store_true')
parser.add_argument('--devices', type=str, default='0,1,2,3')

# 异常检测配置
parser.add_argument('--anomaly_ratio', type=float, default=1.0)

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('='*60)
print('Training Original TimesNet (for MC Dropout)')
print('='*60)
print(f'Dataset: {args.data}')
print(f'Dropout: {args.dropout}')
print(f'Epochs: {args.train_epochs}')
print('='*60)

# 训练
Exp = Exp_Anomaly_Detection
setting = f'{args.model_id}_{args.model}_{args.data}_dropout{args.dropout}'

exp = Exp(args)
print(f'\n>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
exp.train(setting)

print(f'\n>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
exp.test(setting)

print('\n'+'='*60)
print('Training complete!')
print(f'Model saved to: ./checkpoints/{setting}/checkpoint.pth')
print('='*60)
print('\nNext step: Run MC Dropout test')
print(f'python validate_mc_dropout.py --model_path ./checkpoints/{setting}/checkpoint.pth')
print('='*60)

torch.cuda.empty_cache()
