import torch
import torch.nn as nn
import math


class AnomalyBlock(nn.Module):
    def __init__(self, d_model, n_heads, win_size, d_ff, dropout=0.05, sigma_init_factor=5.0):
        super(AnomalyBlock, self).__init__()
        d_keys = d_model // n_heads

        self.n_heads = n_heads
        self.d_keys = d_keys
        self.win_size = win_size

        # Series Branch 的投影层：负责处理 x_series (TimesNet特征)
        self.linear_q = nn.Linear(d_model, n_heads * d_keys)
        self.linear_k = nn.Linear(d_model, n_heads * d_keys)
        self.linear_v = nn.Linear(d_model, n_heads * d_keys)

        # Prior Branch: 使用可学习但数据独立的参数
        # 每个head学习一个固定的sigma值(基于窗口大小)
        # 初始化为合理的局部窗口大小
        init_sigma = torch.ones(n_heads) * (win_size / sigma_init_factor)  # 可配置的初始化因子
        self.prior_sigma = nn.Parameter(init_sigma)

        self.out_layer = nn.Linear(n_heads * d_keys, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, _ = x.shape

        # 1. Series Branch: 使用深层特征生成 Q, K, V
        # 目的：捕捉长距离依赖和周期性 (Global, Data-dependent)
        q = self.linear_q(x).view(B, L, self.n_heads, self.d_keys)
        k = self.linear_k(x).view(B, L, self.n_heads, self.d_keys)
        v = self.linear_v(x).view(B, L, self.n_heads, self.d_keys)

        # 2. 计算 Series Association (Standard Attention)
        scale = self.d_keys ** -0.5
        series_attn = torch.einsum("blhd,bshd->bhls", q, k) * scale
        series_attn = torch.softmax(series_attn, dim=-1)

        # 3. Prior Branch: 使用固定(可学习但数据独立)的 Sigma
        # 目的：提供局部平滑的先验分布，作为对比基准
        # Sigma 不依赖输入数据，只依赖位置关系
        sigma = torch.abs(self.prior_sigma) + 1e-6  # 确保 sigma > 0, [H]
        sigma = sigma.unsqueeze(0).expand(B, -1)  # [B, H]
        prior_attn = self._calculate_prior_association(sigma, L)

        # 4. 重构输出 (使用 Series 的 V)
        weighted_value = torch.einsum("bhls,bshd->blhd", series_attn, v)
        out = weighted_value.reshape(B, L, -1)
        out = self.out_layer(out)

        return out, series_attn, prior_attn

    def _calculate_prior_association(self, sigma, L):
        """
        计算基于固定高斯核的 Prior Association
        Args:
            sigma: [B, H] - 每个head的固定方差参数
            L: int - 序列长度
        Returns:
            prior_attn: [B, H, L, L] - Prior attention矩阵
        """
        B, H = sigma.shape
        device = sigma.device

        # 创建位置索引 [L]
        positions = torch.arange(L, dtype=torch.float32, device=device)

        # 计算位置差异矩阵 [L, L]
        # distances[i, j] = |i - j|
        distances = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))

        # 扩展维度以匹配 batch 和 heads [B, H, L, L]
        distances = distances.unsqueeze(0).unsqueeze(0).expand(B, H, L, L)
        sigma_expanded = sigma.unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

        # 高斯核: exp(-(distance)^2 / (2 * sigma^2))
        prior_attn = torch.exp(-(distances ** 2) / (2 * sigma_expanded ** 2))

        # 归一化为概率分布 (每行和为1)
        prior_attn = prior_attn / (prior_attn.sum(dim=-1, keepdim=True) + 1e-8)

        return prior_attn