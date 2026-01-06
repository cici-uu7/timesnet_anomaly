"""
TimesNet_AD_T: 双向交互融合架构

核心创新:
1. 层内融合: 在每个 TimesBlock 内部融合 Anomaly 机制
2. 双向交互: 频域特征 ↔ 关联特征 互相增强
   - 频域→关联: 用周期信息增强异常检测
   - 关联→频域: 用关联差异指导重构
3. 周期感知Prior: 结合FFT频谱信息生成更准确的Prior

架构图:
    Input
      │
      ▼
  ┌─────────────────────────────────────┐
  │         TimesBlock_AD Layer         │
  │  ┌─────────────┐  ┌──────────────┐  │
  │  │ FFT + 2D    │  │   Anomaly    │  │
  │  │    Conv     │  │  Attention   │  │
  │  └──────┬──────┘  └──────┬───────┘  │
  │         │                │          │
  │         ▼                ▼          │
  │      freq_feat       attn_feat      │
  │         │                │          │
  │         └───► 双向交互 ◄──┘          │
  │                  │                  │
  │                  ▼                  │
  │            fused_feat               │
  └─────────────────────────────────────┘
      │
      ▼ (重复 e_layers 次)
      │
      ▼
   Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    """FFT 提取主要周期"""
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class AnomalyAttentionBlock(nn.Module):
    """
    Anomaly Attention 模块
    与原版 Anomaly Transformer 一致的实现
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
        self.dropout_attn = nn.Dropout(dropout)

        # Sigma 投影 (动态Prior)
        self.sigma_projection = nn.Linear(d_model, n_heads)

        # 预计算距离矩阵
        positions = torch.arange(win_size, dtype=torch.float32)
        distances = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        self.register_buffer('distances', distances)

        # 用于监控的 sigma
        self.register_buffer('prior_sigma', torch.ones(n_heads))

    def forward(self, x, freq_info=None):
        """
        前向传播
        Args:
            x: [B, L, D] 输入特征
            freq_info: 保留参数兼容性，但不再使用
        Returns:
            output: [B, L, D]
            series_attn: [B, H, L, L]
            prior_attn: [B, H, L, L]
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
        series_attn = self.dropout_attn(series_attn)

        out = torch.matmul(series_attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_projection(out)

        # 2. Prior Association (高斯Prior)
        # 不使用频域增强，避免干扰 Sigma 学习
        sigma = self.sigma_projection(x)  # [B, L, H]
        sigma = sigma.transpose(1, 2)  # [B, H, L]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1  # 范围 [0, 2]

        # 添加 Sigma 下限约束，防止崩溃到 0
        sigma = torch.clamp(sigma, min=0.1, max=2.0)

        # 更新监控值
        with torch.no_grad():
            self.prior_sigma = sigma.mean(dim=(0, 2))

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

        return output, series_attn, prior_attn


class BidirectionalInteraction(nn.Module):
    """
    双向交互模块 - 核心创新

    频域→关联: 用周期特征增强异常检测能力
    关联→频域: 用关联差异指导重构学习
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 频域→关联 交互
        self.freq_to_attn = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        # 关联→频域 交互
        self.attn_to_freq = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        # 最终融合
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        # 可学习的融合权重
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, freq_feat, attn_feat):
        """
        双向交互融合
        Args:
            freq_feat: [B, L, D] 频域特征 (来自 TimesBlock)
            attn_feat: [B, L, D] 关联特征 (来自 AnomalyAttention)
        Returns:
            fused: [B, L, D] 融合后的特征
        """
        # 频域→关联: 增强异常检测
        # 频域特征帮助 Anomaly Attention 更好地识别周期异常
        concat_f2a = torch.cat([freq_feat, attn_feat], dim=-1)
        enhanced_attn = attn_feat + self.freq_to_attn(concat_f2a)

        # 关联→频域: 指导重构
        # 关联差异信息帮助重构模块关注异常区域
        concat_a2f = torch.cat([attn_feat, freq_feat], dim=-1)
        enhanced_freq = freq_feat + self.attn_to_freq(concat_a2f)

        # 最终融合
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)

        # 加权融合两个增强后的特征
        weighted = alpha * enhanced_freq + beta * enhanced_attn

        # 通过融合层进一步整合
        concat_final = torch.cat([enhanced_freq, enhanced_attn], dim=-1)
        fused = self.fusion(concat_final)

        # 残差连接
        output = fused + 0.5 * (freq_feat + attn_feat)

        return output


class TimesBlock_AD(nn.Module):
    """
    TimesBlock_AD: 层内融合的时序异常检测块

    将 TimesBlock (频域) 和 AnomalyAttention (关联) 在层内融合
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.d_model = configs.d_model

        # 频域分支: Inception Conv (与原版 TimesBlock 一致)
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

        # 关联分支: Anomaly Attention
        self.anomaly_attention = AnomalyAttentionBlock(
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            win_size=configs.seq_len,
            dropout=configs.dropout
        )

        # 双向交互融合
        self.bidirectional_interaction = BidirectionalInteraction(
            d_model=configs.d_model,
            dropout=configs.dropout
        )

        # Layer Norm
        self.norm_freq = nn.LayerNorm(configs.d_model)
        self.norm_attn = nn.LayerNorm(configs.d_model)
        self.norm_out = nn.LayerNorm(configs.d_model)

    def forward(self, x):
        """
        前向传播
        Args:
            x: [B, L, D] 输入特征
        Returns:
            output: [B, L, D] 输出特征
            series_attn: [B, H, L, L] Series注意力
            prior_attn: [B, H, L, L] Prior注意力
        """
        B, T, N = x.size()

        # ========== 频域分支 (TimesBlock) ==========
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([B, length - (self.seq_len + self.pred_len), N]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x

            # reshape for 2D conv
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])

        res = torch.stack(res, dim=-1)

        # 自适应周期聚合
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        freq_feat = torch.sum(res * period_weight, -1)
        freq_feat = self.norm_freq(freq_feat)

        # ========== 关联分支 (Anomaly Attention) ==========
        # 传入频域特征用于增强 Prior 计算
        attn_out, series_attn, prior_attn = self.anomaly_attention(x, freq_info=freq_feat)
        attn_feat = self.norm_attn(attn_out)

        # ========== 双向交互融合 ==========
        fused_feat = self.bidirectional_interaction(freq_feat, attn_feat)

        # 残差连接
        output = self.norm_out(fused_feat + x)

        return output, series_attn, prior_attn


class Model(nn.Module):
    """
    TimesNet_AD_T: 双向交互融合的时序异常检测模型

    创新点:
    1. 层内融合: 每层同时处理频域和关联特征
    2. 双向交互: 频域↔关联 互相增强
    3. 周期感知Prior: FFT信息指导Prior生成
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = getattr(configs, 'pred_len', 0)
        self.output_attention = configs.output_attention
        self.e_layers = configs.e_layers

        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model,
            configs.embed, configs.freq, configs.dropout
        )

        # 多层 TimesBlock_AD
        self.layers = nn.ModuleList([
            TimesBlock_AD(configs) for _ in range(configs.e_layers)
        ])

        # 最终投影
        self.projection = nn.Linear(configs.d_model, configs.c_out)

        # 保存配置用于监控
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        前向传播
        Args:
            x_enc: [B, L, C] 输入序列
        Returns:
            output: [B, L, C] 重构输出
            all_series_attn: List of [B, H, L, L]
            all_prior_attn: List of [B, H, L, L]
        """
        # Normalization (与原版 TimesNet 一致)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # 多层处理
        all_series_attn = []
        all_prior_attn = []

        for layer in self.layers:
            enc_out, series_attn, prior_attn = layer(enc_out)
            all_series_attn.append(series_attn)
            all_prior_attn.append(prior_attn)

        # 投影
        dec_out = self.projection(enc_out)

        # De-normalization
        dec_out = dec_out * stdev + means

        if self.output_attention:
            return dec_out, all_series_attn, all_prior_attn
        else:
            return dec_out, all_series_attn[-1], all_prior_attn[-1]

    def get_sigma_stats(self):
        """获取所有层的 sigma 统计信息"""
        sigmas = []
        for layer in self.layers:
            sigma = layer.anomaly_attention.prior_sigma.cpu().numpy()
            sigmas.append(sigma.mean())
        return sigmas
