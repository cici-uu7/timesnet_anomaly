import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.window_size = win_size
        # 初始化距离矩阵，用于计算高斯核
        self.distances = torch.zeros((self.window_size, self.window_size)).cuda()
        for i in range(self.window_size):
            for j in range(self.window_size):
                self.distances[i][j] = abs(i - j)

    def forward(self, queries, keys, values, sigma, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        # 1. 计算 Standard Self-Attention (Series Association)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = torch.triu(torch.ones(L, S), diagonal=1).bool().to(queries.device)
            scores.masked_fill_(attn_mask, -np.inf)

        attn = scale * scores
        series_attn_probs = self.dropout(torch.softmax(attn, dim=-1))

        # 2. 计算 Gaussian Kernel (Prior Association)
        # sigma: [B, L, H] -> [B, H, L, 1] 为了广播机制
        sigma = sigma.transpose(1, 2).contiguous()  # [B, H, L]
        window_size = self.window_size
        self.distances = self.distances.to(queries.device)

        # 扩展维度以匹配 [B, H, L, L]
        # distances: [L, L] -> [1, 1, L, L] -> [B, H, L, L]
        dist = self.distances.unsqueeze(0).unsqueeze(0).repeat(B, H, 1, 1)

        # sigma: [B, H, L] -> [B, H, L, 1]
        sigma_expanded = sigma.unsqueeze(-1)

        # 高斯分布公式计算
        prior_attn_probs = 1.0 / (math.sqrt(2 * math.pi) * sigma_expanded) * torch.exp(
            -dist ** 2 / (2 * sigma_expanded ** 2))

        # 归一化 Prior
        prior_attn_probs = prior_attn_probs / (prior_attn_probs.sum(-1, keepdim=True) + 1e-10)

        # 3. 计算输出
        V = values
        output = torch.einsum("bhls,bshd->blhd", series_attn_probs, V)

        return output, series_attn_probs, prior_attn_probs


class AnomalyBlock(nn.Module):
    def __init__(self, d_model, n_heads, win_size, d_ff, dropout=0.05):
        super(AnomalyBlock, self).__init__()
        self.attention = AnomalyAttention(win_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 生成 Q, K, V 和 Sigma
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_sigma = nn.Linear(d_model, n_heads)  # 为每个 Head 学习一个 sigma

        self.n_heads = n_heads
        self.d_keys = d_model // n_heads

        # FFN
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.activation = F.gelu

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape

        # 投影生成 Q, K, V
        q = self.linear_q(x).view(B, L, self.n_heads, self.d_keys)
        k = self.linear_k(x).view(B, L, self.n_heads, self.d_keys)
        v = self.linear_v(x).view(B, L, self.n_heads, self.d_keys)

        # 生成 Sigma (确保为正数)
        sigma = torch.sigmoid(self.linear_sigma(x)) + 1e-6

        # 计算 Attention
        new_x, series_attn, prior_attn = self.attention(q, k, v, sigma, attn_mask=None)
        new_x = new_x.reshape(B, L, D)

        # 残差连接 1
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)

        # FFN 部分
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # 残差连接 2
        return self.norm2(x + y), series_attn, prior_attn