import torch
import torch.nn as nn
import math


class AnomalyBlock(nn.Module):
    def __init__(self, d_model, n_heads, win_size, d_ff, dropout=0.05):
        super(AnomalyBlock, self).__init__()
        d_keys = d_model // n_heads

        self.n_heads = n_heads
        self.d_keys = d_keys

        # Series Branch 的投影层：负责处理 x_series (TimesNet特征)
        self.linear_q = nn.Linear(d_model, n_heads * d_keys)
        self.linear_k = nn.Linear(d_model, n_heads * d_keys)
        self.linear_v = nn.Linear(d_model, n_heads * d_keys)

        # Prior Branch 的投影层：负责处理 x_prior (原始Embedding)
        self.linear_sigma = nn.Linear(d_model, n_heads)

        self.out_layer = nn.Linear(n_heads * d_keys, d_model)
        self.dropout = nn.Dropout(dropout)

    # 核心修改点：接收 x_series 和 x_prior 两个输入
    def forward(self, x_series, x_prior):
        B, L, _ = x_series.shape

        # 1. Series Branch: 使用 TimesNet 深层特征生成 Q, K, V
        # 目的：捕捉长距离依赖和周期性 (Global)
        q = self.linear_q(x_series).view(B, L, self.n_heads, self.d_keys)
        k = self.linear_k(x_series).view(B, L, self.n_heads, self.d_keys)
        v = self.linear_v(x_series).view(B, L, self.n_heads, self.d_keys)

        # 2. Prior Branch: 使用 原始/浅层特征生成 Sigma
        # 目的：专注于局部连续性 (Local)，不受深层特征干扰
        sigma = self.linear_sigma(x_prior).view(B, L, self.n_heads)
        sigma = torch.sigmoid(sigma) + 1e-6  # 确保 sigma > 0

        # 3. 计算 Series Association (Standard Attention)
        scale = self.d_keys ** -0.5
        series_attn = torch.einsum("blhd,bshd->bhls", q, k) * scale
        series_attn = torch.softmax(series_attn, dim=-1)

        # 4. 计算 Prior Association (Gaussian Kernel)
        sigma = sigma.permute(0, 2, 1).contiguous().float()  # [B, H, L]
        prior_attn = self._calculate_prior_association(sigma, L)

        # 5. 重构输出 (使用 Series 的 V)
        weighted_value = torch.einsum("bhls,bshd->blhd", series_attn, v)
        out = weighted_value.reshape(B, L, -1)
        out = self.out_layer(out)

        return out, series_attn, prior_attn

    def _calculate_prior_association(self, sigma, L):
        # 高斯核计算逻辑
        # 这里的逻辑是动态生成的，虽然有一点点算力浪费，但适应性强（不用管 Window Size 变化）
        B, H, _ = sigma.shape
        x_indices = torch.arange(L).to(sigma.device).unsqueeze(0).unsqueeze(0).expand(B, H, L)
        mean = x_indices

        # 计算高斯分布：exp(-(x - mu)^2 / 2sigma^2)
        # 注意 unsqueeze 是为了广播：[B, H, L, 1] vs [B, H, 1, L] -> [B, H, L, L]
        gaussian = torch.exp(-((x_indices.unsqueeze(-1) - mean.unsqueeze(2)) ** 2) / (2 * (sigma.unsqueeze(2) ** 2)))

        # 归一化，使其成为概率分布 (Sum=1)
        gaussian = gaussian / (torch.sum(gaussian, dim=-1).unsqueeze(-1) + 1e-5)
        return gaussian