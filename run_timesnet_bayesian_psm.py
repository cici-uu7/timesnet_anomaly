"""
运行脚本：TimesNet + MC Dropout (PSM数据集)

使用方法：
python run_timesnet_bayesian_psm.py
"""

import argparse
import torch
from exp.exp_timesnet_ad_bayesian import Exp_TimesNet_AD_Bayesian
import random
import numpy as np

# 固定随机种子
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='TimesNet with MC Dropout for Anomaly Detection')

# 基础配置
parser.add_argument('--task_name', type=str, default='anomaly_detection')
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, default='PSM', help='model id')
parser.add_argument('--model', type=str, default='TimesNet_AD_Bayesian')

# 数据配置
parser.add_argument('--data', type=str, default='PSM', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/PSM', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='train.csv', help='data csv file')
parser.add_argument('--features', type=str, default='M', help='M: multivariate')
parser.add_argument('--target', type=str, default='OT', help='target feature')
parser.add_argument('--freq', type=str, default='h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# 时序配置
parser.add_argument('--seq_len', type=int, default=100, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=0, help='prediction sequence length')

# 模型配置
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=25, help='encoder input size (PSM: 25)')
parser.add_argument('--dec_in', type=int, default=25, help='decoder input size')
parser.add_argument('--c_out', type=int, default=25, help='output size')
parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
parser.add_argument('--activation', type=str, default='gelu')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention')

# MC Dropout配置
parser.add_argument('--mc_samples', type=int, default=5, help='MC Dropout samples')
parser.add_argument('--uncertainty_weight', type=float, default=0.5, help='uncertainty weight')
parser.add_argument('--score_strategy', type=str, default='weighted',
                    choices=['weighted', 'additive', 'multiplicative'])

# 训练配置
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

# GPU配置
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--gpu_type', type=str, default='cuda', choices=['cuda', 'mps'])
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus')
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids')

# 异常检测配置
parser.add_argument('--anomaly_ratio', type=float, default=1.0, help='anomaly ratio')

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

# 运行实验
Exp = Exp_TimesNet_AD_Bayesian

for ii in range(args.itr):
    setting = f'{args.model_id}_{args.model}_{args.data}_mc{args.mc_samples}_uw{args.uncertainty_weight}'

    exp = Exp(args)

    if args.is_training:
        print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.train(setting)
        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting)
    else:
        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)

    torch.cuda.empty_cache()
