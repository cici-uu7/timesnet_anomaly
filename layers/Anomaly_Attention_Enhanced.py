"""
增强版 Anomaly Attention Block
支持：
1. 动态Prior（混合静态高斯核 + 数据驱动Prior）
2. 多头注意力
3. 更强的特征提取能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DynamicPriorNetwork(nn.Module):
    """
    动态Prior生成网络
    学习数据的"正常模式"作为Prior的一部分
    """
    def __init__(self, d_model, n_heads, win_size, dropout=0.05):
        super(DynamicPriorNetwork, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.win_size = win_size

        # 轻量级卷积网络提取局部模式
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)

        # 全局平均池化 + MLP生成Prior参数
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_heads)
        )

    def forward(self, x):
        """
        输入: x [B, L, D]
        输出: dynamic_prior_weights [B, H] - 每个头的动态权重
        """
        B, L, D = x.shape

        # 转置为 [B, D, L] 用于卷积
        x_conv = x.transpose(1, 2)

        # 多尺度特征提取
        feat1 = F.gelu(self.conv1(x_conv))  # [B, D, L]
        feat2 = F.gelu(self.conv2(x_conv))  # [B, D, L]

        # 全局池化
        feat1_pool = feat1.mean(dim=-1)  # [B, D]
        feat2_pool = feat2.mean(dim=-1)  # [B, D]

        # 拼接并生成动态权重
        feat_concat = torch.cat([feat1_pool, feat2_pool], dim=-1)  # [B, 2D]
        dynamic_weights = torch.sigmoid(self.mlp(feat_concat))  # [B, H]

        return dynamic_weights


class AnomalyBlockEnhanced(nn.Module):
    """
    增强版 Anomaly Block
    特性：
    1. 混合Prior（静态高斯核 + 动态Prior）
    2. 多头注意力机制
    3. 残差连接
    """
    def __init__(self, d_model, n_heads, win_size, d_ff, dropout=0.05,
                 sigma_init_factor=5.0, dynamic_prior=True):
        super(AnomalyBlockEnhanced, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.win_size = win_size
        self.dynamic_prior = dynamic_prior

        # Series Branch: Multi-head Self-Attention
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout_attn = nn.Dropout(dropout)

        # Prior Branch: 静态高斯核参数（数据独立）
        init_sigma = torch.ones(n_heads) * (win_size / sigma_init_factor)
        self.prior_sigma = nn.Parameter(init_sigma)

        # Prior Branch: 动态Prior生成网络（可选）
        if dynamic_prior:
            self.dynamic_prior_net = DynamicPriorNetwork(d_model, n_heads, win_size, dropout)
            # 混合权重：静态 vs 动态
            self.static_dynamic_balance = nn.Parameter(torch.tensor(0.5))

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

    def _calculate_static_prior(self, L, device):
        """
        计算静态Prior（高斯核）
        返回: [H, L, L]
        """
        positions = torch.arange(L, dtype=torch.float32, device=device)
        distances = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))  # [L, L]

        # 为每个头计算不同的高斯核
        sigma = torch.clamp(self.prior_sigma, min=1e-2)  # [H]
        sigma_expanded = sigma.view(-1, 1, 1)  # [H, 1, 1]
        distances_expanded = distances.unsqueeze(0)  # [1, L, L]

        # 高斯核: exp(-d^2 / (2*sigma^2))
        prior_attn = torch.exp(-(distances_expanded ** 2) / (2 * sigma_expanded ** 2))

        # 归一化
        prior_attn = prior_attn / (prior_attn.sum(dim=-1, keepdim=True) + 1e-8)

        return prior_attn  # [H, L, L]

    def _calculate_dynamic_prior_adjustment(self, x, static_prior):
        """
        计算动态Prior调整
        输入:
            x: [B, L, D]
            static_prior: [H, L, L]
        返回: [B, H, L, L]
        """
        B, L, D = x.shape
        H = self.n_heads

        # 生成动态权重 [B, H]
        dynamic_weights = self.dynamic_prior_net(x)  # [B, H]

        # 混合静态和动态Prior
        # 静态权重
        static_weight = torch.sigmoid(self.static_dynamic_balance)
        dynamic_weight = 1.0 - static_weight

        # 扩展静态Prior [H, L, L] -> [B, H, L, L]
        static_prior_expanded = static_prior.unsqueeze(0).expand(B, -1, -1, -1)

        # 动态调整：根据数据特征微调Prior
        # dynamic_weights [B, H] 控制每个头的Prior强度
        dynamic_adjustment = dynamic_weights.view(B, H, 1, 1)  # [B, H, 1, 1]

        # 混合Prior = 静态部分 + 动态调整部分
        mixed_prior = static_weight * static_prior_expanded + \
                      dynamic_weight * dynamic_adjustment * static_prior_expanded

        # 重新归一化
        mixed_prior = mixed_prior / (mixed_prior.sum(dim=-1, keepdim=True) + 1e-8)

        return mixed_prior  # [B, H, L, L]

    def _calculate_series_association(self, x):
        """
        计算Series Association（Multi-head Self-Attention）
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

        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # [B, H, L, L]
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
            series_attn [B, H, L, L] - Series注意力
            prior_attn [B, H, L, L] or [H, L, L] - Prior注意力
        """
        B, L, D = x.shape

        # 1. 计算Series Association
        series_out, series_attn = self._calculate_series_association(x)  # [B, L, D], [B, H, L, L]

        # 2. 计算Prior Association
        static_prior = self._calculate_static_prior(L, x.device)  # [H, L, L]

        if self.dynamic_prior:
            # 混合Prior
            prior_attn = self._calculate_dynamic_prior_adjustment(x, static_prior)  # [B, H, L, L]
        else:
            # 仅静态Prior
            prior_attn = static_prior  # [H, L, L]

        # 3. 残差连接 + Layer Norm
        x = self.norm1(x + self.dropout(series_out))

        # 4. Feed-Forward Network
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x, series_attn, prior_attn
