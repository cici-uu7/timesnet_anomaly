"""
TimesNet_AD_V2: 最后层融合架构 (Last-Layer Fusion)

核心思想:
1. 前 N-1 层使用纯 TimesNet (保证重构能力)
2. 最后一层引入 AnomalyAttention (添加关联差异检测)
3. 两种异常信号独立计算，最后加权组合

这种设计的优势:
- TimesNet 的频域特征提取不受干扰
- AnomalyAttention 利用已经提取好的高层语义特征
- 避免两种机制在中间层互相干扰

架构图:
    Input
      │
      ▼
  ┌─────────────────┐
  │   TimesBlock    │  Layer 1
  └─────────────────┘
      │
      ▼
  ┌─────────────────┐
  │   TimesBlock    │  Layer 2
  └─────────────────┘
      │
      ▼
  ┌─────────────────────────────────┐
  │        Last Layer               │
  │  ┌───────────┐  ┌────────────┐  │
  │  │TimesBlock │  │  Anomaly   │  │
  │  │           │  │ Attention  │  │
  │  └─────┬─────┘  └─────┬──────┘  │
  │        │              │         │
  │        ▼              ▼         │
  │    重构输出       关联差异       │
  └─────────────────────────────────┘
      │              │
      ▼              ▼
   rec_loss    assoc_discrepancy
      │              │
      └──────┬───────┘
             ▼
      anomaly_score = α * rec + β * assoc
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    """FFT 提取主要周期 - 与原版 TimesNet 完全一致"""
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    """
    原版 TimesBlock - 不做任何修改
    保持 TimesNet 的原始重构能力
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([B, length - (self.seq_len + self.pred_len), N]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x

            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])

        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)

        # 残差连接
        res = res + x
        return res


class AnomalyAttention(nn.Module):
    """
    Anomaly Attention 模块 - 基于 Anomaly Transformer
    仅用于关联差异计算，不参与重构
    """
    def __init__(self, d_model, n_heads, win_size, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.win_size = win_size

        # Multi-head Self-Attention
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Sigma 投影 (动态Prior)
        self.sigma_projection = nn.Linear(d_model, n_heads)

        # 预计算距离矩阵
        positions = torch.arange(win_size, dtype=torch.float32)
        distances = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        self.register_buffer('distances', distances)

    def forward(self, x):
        """
        Args:
            x: [B, L, D] 输入特征
        Returns:
            output: [B, L, D] 注意力输出
            series_attn: [B, H, L, L] Series 注意力
            prior_attn: [B, H, L, L] Prior 注意力
        """
        B, L, D = x.shape
        H = self.n_heads
        d_k = self.d_k

        # 1. Series Association (Multi-head Self-Attention)
        Q = self.query_projection(x).view(B, L, H, d_k).transpose(1, 2)
        K = self.key_projection(x).view(B, L, H, d_k).transpose(1, 2)
        V = self.value_projection(x).view(B, L, H, d_k).transpose(1, 2)

        scale = 1.0 / math.sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        series_attn = F.softmax(scores, dim=-1)
        series_attn = self.dropout(series_attn)

        out = torch.matmul(series_attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_projection(out)

        # 2. Prior Association (高斯Prior)
        sigma = self.sigma_projection(x)  # [B, L, H]
        sigma = sigma.transpose(1, 2)  # [B, H, L]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1  # 范围 [0, 2]
        sigma = torch.clamp(sigma, min=0.1, max=2.0)  # 防止崩溃

        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, L)  # [B, H, L, L]

        # 距离矩阵
        if L != self.win_size:
            positions = torch.arange(L, dtype=torch.float32, device=x.device)
            distances = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        else:
            distances = self.distances

        distances = distances.unsqueeze(0).unsqueeze(0).repeat(B, H, 1, 1)

        # 高斯 Prior
        prior_attn = 1.0 / (math.sqrt(2 * math.pi) * sigma) * \
                     torch.exp(-distances ** 2 / (2 * sigma ** 2))

        # 归一化 Prior
        prior_attn = prior_attn / (prior_attn.sum(dim=-1, keepdim=True) + 1e-8)

        return output, series_attn, prior_attn


class Model(nn.Module):
    """
    TimesNet_AD_V2: 最后层融合架构

    设计原则:
    1. 保持 TimesNet 的重构能力 (前 N-1 层纯 TimesBlock)
    2. 仅在最后一层添加关联差异检测
    3. 两种异常信号独立计算，不互相干扰
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = getattr(configs, 'pred_len', 0)
        self.output_attention = configs.output_attention
        self.e_layers = configs.e_layers
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads

        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model,
            configs.embed, configs.freq, configs.dropout
        )

        # 前 N-1 层: 纯 TimesBlock (保持重构能力)
        self.times_layers = nn.ModuleList([
            TimesBlock(configs) for _ in range(configs.e_layers)
        ])

        # Layer Norm for each TimesBlock
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(configs.d_model) for _ in range(configs.e_layers)
        ])

        # 最后一层: Anomaly Attention (关联差异检测)
        self.anomaly_attention = AnomalyAttention(
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            win_size=configs.seq_len,
            dropout=configs.dropout
        )
        self.attn_norm = nn.LayerNorm(configs.d_model)

        # 投影层
        self.projection = nn.Linear(configs.d_model, configs.c_out)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        前向传播
        Args:
            x_enc: [B, L, C] 输入序列
        Returns:
            dec_out: [B, L, C] 重构输出
            series_attn: [B, H, L, L] Series注意力 (来自最后层)
            prior_attn: [B, H, L, L] Prior注意力 (来自最后层)
        """
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # 所有层使用 TimesBlock (保持重构能力)
        for i in range(self.e_layers):
            enc_out = self.times_layers[i](enc_out)
            enc_out = self.layer_norms[i](enc_out)

        # 最后: Anomaly Attention (仅用于关联差异，不改变 enc_out)
        # 使用最后一层的特征计算关联差异
        attn_out, series_attn, prior_attn = self.anomaly_attention(enc_out)
        # 注意: attn_out 不参与重构，仅用于计算关联差异

        # 投影 (仅使用 TimesBlock 的输出)
        dec_out = self.projection(enc_out)

        # De-normalization
        dec_out = dec_out * stdev + means

        if self.output_attention:
            return dec_out, [series_attn], [prior_attn]
        else:
            return dec_out, series_attn, prior_attn

    def get_sigma_stats(self):
        """获取 Anomaly Attention 的 sigma 统计"""
        return ["N/A"]  # 只有一层 AnomalyAttention
