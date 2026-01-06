"""
增强版 Anomaly Attention Block
基于原版 Anomaly Transformer 的实现方式改进：
1. 动态 Sigma：从数据投影生成，范围约束在 [0, 2]（与原版一致）
2. 多头注意力
3. 与原版一致的 Prior 计算方式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AnomalyBlockEnhanced(nn.Module):
    """
    增强版 Anomaly Block（与原版 Anomaly Transformer 对齐）

    关键改进：
    1. Sigma 从数据动态投影，而非固定参数
    2. Sigma 范围约束在 [0, 2]，通过 sigmoid + pow(3, x) - 1 实现
    3. Prior 计算方式与原版完全一致
    """
    def __init__(self, d_model, n_heads, win_size, d_ff, dropout=0.05,
                 sigma_init_factor=5.0, dynamic_prior=True):
        super(AnomalyBlockEnhanced, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.win_size = win_size

        # sigma_init_factor 和 dynamic_prior 参数保留兼容性但不再使用
        # 新实现始终使用动态 sigma

        # Series Branch: Multi-head Self-Attention
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout_attn = nn.Dropout(dropout)

        # Prior Branch: Sigma 投影层（与原版 Anomaly Transformer 一致）
        # 从数据投影生成 sigma，而非固定可学习参数
        self.sigma_projection = nn.Linear(d_model, n_heads)

        # 保留 prior_sigma 用于监控（但不参与计算）
        # 这里存储的是每个 batch 的 sigma 均值，用于诊断
        self.register_buffer('prior_sigma', torch.ones(n_heads))

        # 预计算距离矩阵
        self.register_buffer('distances', self._precompute_distances(win_size))

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _precompute_distances(self, win_size):
        """预计算位置距离矩阵 [L, L]"""
        positions = torch.arange(win_size, dtype=torch.float32)
        distances = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        return distances

    def _calculate_prior(self, x, series_attn):
        """
        计算 Prior Association（与原版 Anomaly Transformer 一致）

        关键点：
        1. Sigma 从数据投影：sigma_projection(x)
        2. Sigma 范围约束：sigmoid(sigma * 5) -> pow(3, sigma) - 1 -> 范围 [0, 2]
        3. 高斯 Prior：1/(sqrt(2π)σ) * exp(-d²/(2σ²))

        输入:
            x: [B, L, D] - 输入特征
            series_attn: [B, H, L, L] - Series 注意力（用于获取形状）
        返回:
            prior_attn: [B, H, L, L] - Prior 注意力
            sigma: [B, H, L] - 用于计算的 sigma 值
        """
        B, L, D = x.shape
        H = self.n_heads

        # 1. 从数据投影 Sigma [B, L, H]
        sigma = self.sigma_projection(x)  # [B, L, H]
        sigma = sigma.transpose(1, 2)     # [B, H, L]

        # 2. Sigma 范围约束（与原版一致）
        # sigmoid(sigma * 5) 范围 [0, 1]
        # pow(3, x) - 1 范围 [0, 2]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1   # 范围约 [0, 2]

        # 更新 prior_sigma buffer 用于监控
        with torch.no_grad():
            self.prior_sigma = sigma.mean(dim=(0, 2))  # [H]

        # 3. 扩展维度用于广播（与原版一致）
        # 关键：sigma 在最后一维重复，使每一行的 sigma 值相同
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, L)  # [B, H, L, L]

        # 4. 获取距离矩阵并扩展
        # distances: [L, L] -> [1, 1, L, L]
        if L != self.win_size:
            # 动态计算距离矩阵（当序列长度不同时）
            positions = torch.arange(L, dtype=torch.float32, device=x.device)
            distances = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        else:
            distances = self.distances

        distances = distances.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
        distances = distances.repeat(B, H, 1, 1)         # [B, H, L, L]

        # 5. 计算高斯 Prior（与原版一致）
        # prior = 1/(sqrt(2π)σ) * exp(-d²/(2σ²))
        prior_attn = 1.0 / (math.sqrt(2 * math.pi) * sigma) * \
                     torch.exp(-distances ** 2 / (2 * sigma ** 2))

        # 注意：原版不对 Prior 归一化，保持这个特性
        # 归一化在 KL 散度计算时进行

        return prior_attn, sigma  # [B, H, L, L], [B, H, L, L]（sigma已经repeat过了）

    def _calculate_series_association(self, x):
        """
        计算 Series Association（Multi-head Self-Attention）
        输入: x [B, L, D]
        返回: output [B, L, D], attention [B, H, L, L]
        """
        B, L, D = x.shape
        H = self.n_heads
        d_k = self.d_k

        # 投影
        Q = self.query_projection(x).view(B, L, H, d_k).transpose(1, 2)  # [B, H, L, d_k]
        K = self.key_projection(x).view(B, L, H, d_k).transpose(1, 2)    # [B, H, L, d_k]
        V = self.value_projection(x).view(B, L, H, d_k).transpose(1, 2)  # [B, H, L, d_k]

        # 注意力分数（缩放点积）
        scale = 1.0 / math.sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, L, L]

        # Softmax 归一化
        series_attn = F.softmax(scores, dim=-1)  # [B, H, L, L]
        series_attn = self.dropout_attn(series_attn)

        # 加权求和
        out = torch.matmul(series_attn, V)  # [B, H, L, d_k]
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        out = self.out_projection(out)

        return out, series_attn

    def forward(self, x):
        """
        前向传播
        输入: x [B, L, D]
        返回:
            output [B, L, D] - 增强后的特征
            series_attn [B, H, L, L] - Series 注意力（已归一化）
            prior_attn [B, H, L, L] - Prior 注意力（未归一化，与原版一致）
        """
        B, L, D = x.shape

        # 1. 计算 Series Association
        series_out, series_attn = self._calculate_series_association(x)

        # 2. 计算 Prior Association（动态 Sigma）
        prior_attn, sigma = self._calculate_prior(x, series_attn)

        # 3. 残差连接 + Layer Norm
        x = self.norm1(x + self.dropout(series_out))

        # 4. Feed-Forward Network
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x, series_attn, prior_attn
