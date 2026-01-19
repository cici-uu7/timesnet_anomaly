"""
TimesNetPro: TimesNet with Adaptive Period Attention

核心改进：自适应周期注意力 (Adaptive Period Attention)

问题分析：
- 原版 TimesNet 使用简单的 softmax(FFT幅度) 作为周期权重
- 假设所有 Top-k 周期同等重要，直接相加会"稀释"异常信号
- 异常往往只破坏某一个特定周期，其他周期可能正常

解决方案：
- 引入可学习的注意力机制，让模型动态判断每个周期的"信息量"
- 结合 FFT 权重（频域先验）和学习的注意力权重（数据驱动）
- 使用轻量级的注意力模块，不增加太多计算开销

优势：
- 改动最小，逻辑最硬
- 保持 TimesNet 的核心架构不变
- 适用于异常检测任务，能更好地捕捉周期异常
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.TimesNet import FFT_for_Period
from layers.Conv_Blocks import Inception_Block_V1


class AdaptivePeriodAttention(nn.Module):
    """
    自适应周期注意力模块
    
    对每个周期的特征进行编码，学习其重要性权重
    结合 FFT 权重（频域先验）和学习的注意力权重
    """
    def __init__(self, d_model, d_attn=None, dropout=0.1):
        """
        Args:
            d_model: 特征维度
            d_attn: 注意力隐藏维度（默认 d_model // 4）
            dropout: Dropout 比率
        """
        super(AdaptivePeriodAttention, self).__init__()
        self.d_model = d_model
        self.d_attn = d_attn or (d_model // 4)
        
        # 周期特征编码器：将每个周期的特征编码为固定维度
        # 使用全局池化 + MLP 来提取周期级别的特征
        self.period_encoder = nn.Sequential(
            nn.Linear(d_model, self.d_attn),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_attn, self.d_attn)
        )
        
        # Query 生成器：从输入序列生成查询向量
        # 用于判断哪些周期对当前样本更重要
        self.query_proj = nn.Linear(d_model, self.d_attn)
        
        # 温度参数（可学习），用于控制注意力分布的锐度
        self.temperature = nn.Parameter(torch.ones(1))
        
        # 融合权重：平衡 FFT 权重和学习的注意力权重
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始值：两者各占一半
        
        # 注意：在 TimesNet 中，输入特征维度 N 通常等于 d_model
        # 所以 period_global 的维度应该是 [B, K, d_model]，不需要额外投影
        
    def forward(self, period_features, fft_weights, x_input):
        """
        Args:
            period_features: [B, T, N, K] K 个周期的特征
            fft_weights: [B, K] FFT 计算的原始权重
            x_input: [B, T, N] 原始输入，用于生成 query
        
        Returns:
            attention_weights: [B, T, N, K] 最终的周期注意力权重
        """
        B, T, N, K = period_features.shape
        
        # 1. 对每个周期的特征进行编码
        # period_features: [B, T, N, K] -> [B, K, T, N]
        period_feat = period_features.permute(0, 3, 1, 2).contiguous()  # [B, K, T, N]
        
        # 全局池化：提取每个周期的全局特征
        # 对每个周期，在时间维度上进行全局平均池化
        # [B, K, T, N] -> [B, K, N] (在 TimesNet 中，N == d_model)
        period_global = period_feat.mean(dim=2)  # [B, K, N] 对时间维度求平均
        
        # 编码周期特征: [B, K, d_model] -> [B, K, d_attn]
        period_encoded = self.period_encoder(period_global)  # [B, K, d_attn]
        
        # 2. 从输入生成 Query
        # 使用输入序列的全局特征作为 query
        x_global = x_input.mean(dim=1)  # [B, N] -> 对时间维度求平均
        query = self.query_proj(x_global)  # [B, d_attn]
        query = query.unsqueeze(1)  # [B, 1, d_attn]
        
        # 3. 计算注意力分数
        # query: [B, 1, d_attn], period_encoded: [B, K, d_attn]
        attn_scores = torch.matmul(query, period_encoded.transpose(1, 2))  # [B, 1, K]
        attn_scores = attn_scores / (self.temperature + 1e-8)  # 温度缩放
        attn_scores = attn_scores.squeeze(1)  # [B, K]
        
        # 4. 结合 FFT 权重和学习的注意力权重
        # FFT 权重归一化
        fft_weights_norm = F.softmax(fft_weights, dim=1)  # [B, K]
        
        # 学习的注意力权重归一化
        learned_weights = F.softmax(attn_scores, dim=1)  # [B, K]
        
        # 融合：alpha 控制两者的平衡（可学习）
        alpha = torch.sigmoid(self.alpha)  # 确保在 [0, 1] 之间
        fused_weights = alpha * learned_weights + (1 - alpha) * fft_weights_norm  # [B, K]
        
        # 5. 扩展到 [B, T, N, K] 维度
        fused_weights = fused_weights.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, K]
        attention_weights = fused_weights.repeat(1, T, N, 1)  # [B, T, N, K]
        
        return attention_weights


class TimesBlockPro(nn.Module):
    """
    TimesBlock with Adaptive Period Attention
    
    改进点：
    1. 使用自适应周期注意力替代简单的 FFT 权重聚合
    2. 保持 TimesNet 的核心架构（2D Conv）不变
    3. 可学习的注意力机制能更好地捕捉周期异常
    """
    def __init__(self, configs):
        super(TimesBlockPro, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.d_model = configs.d_model
        
        # 2D 卷积块（与原版一致）
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )
        
        # 自适应周期注意力模块
        dropout_rate = getattr(configs, 'dropout', 0.1)
        self.period_attention = AdaptivePeriodAttention(
            d_model=configs.d_model,
            dropout=dropout_rate
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, T, N] 输入序列
        
        Returns:
            res: [B, T, N] 输出特征
        """
        B, T, N = x.size()

        # ============================================================
        # 🧪 消融实验模式：强制退化回原版 TimesNet 逻辑
        # 仅使用 FFT 幅度 period_weight 作为周期聚合权重
        # ============================================================

        # 1. FFT 提取周期和原始权重（幅度）
        period_list, period_weight = FFT_for_Period(x, self.k)

        # 2. 对每个周期进行 2D 卷积处理（与原版 TimesBlock 完全一致）
        res = []
        for i in range(self.k):
            period = period_list[i]

            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([B, length - (self.seq_len + self.pred_len), N]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x

            # reshape: [B, T, N] -> [B, N, H, W]
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv
            out = self.conv(out)
            # reshape back: [B, N, H, W] -> [B, T, N]
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])

        # 3. 堆叠所有周期特征（严格复刻原版聚合的张量形状）
        # list[[B, L, N]] -> [B, K, L, N]
        res = torch.stack(res, dim=1)

        # 4. 使用 FFT 幅度作为权重 (原版 TimesNet 逻辑)
        # period_weight: [B, K] -> Softmax 归一化 -> [B, K, 1, 1]
        weights = F.softmax(period_weight, dim=1).unsqueeze(-1).unsqueeze(-1)

        # 调试：确认广播形状
        # 预期 weights: [B, K, 1, 1], res: [B, K, L, N]
        print("TimesBlockPro fusion shapes -> weights:", weights.shape, "res:", res.shape)

        # 5. 加权融合（dim=1 是 K 维度）: [B, K, L, N] -> [B, L, N]
        res = torch.sum(res * weights, dim=1)

        # 6. 残差连接
        res = res + x

        return res


class Model(nn.Module):
    """
    TimesNetPro: TimesNet with Adaptive Period Attention
    
    使用方法：
    与原版 TimesNet 完全兼容，可以直接替换使用
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        
        # 使用改进的 TimesBlockPro
        self.model = nn.ModuleList([TimesBlockPro(configs)
                                    for _ in range(configs.e_layers)])
        
        # Embedding 层
        from layers.Embed import DataEmbedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model,
                                          configs.embed, configs.freq,
                                          configs.dropout)
        
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        
        # Projection 层
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)
        
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        
        # TimesNetPro blocks
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        
        # Projection
        dec_out = self.projection(enc_out)
        
        # De-normalization
        dec_out = dec_out.mul(
            (stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
            (means[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1)))
        return dec_out
    
    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc.sub(means)
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc.div(stdev)
        
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # TimesNetPro blocks
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        
        # Projection
        dec_out = self.projection(enc_out)
        
        # De-normalization
        dec_out = dec_out.mul(
            (stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.seq_len + self.pred_len, 1)))
        dec_out = dec_out.add(
            (means[:, 0, :].unsqueeze(1).repeat(
                1, self.seq_len + self.pred_len, 1)))
        return dec_out
    
    def anomaly_detection(self, x_enc):
        """
        异常检测前向传播
        """
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)
        
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        
        # TimesNetPro blocks
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        
        # Projection
        dec_out = self.projection(enc_out)
        
        # De-normalization
        dec_out = dec_out.mul(
            (stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
            (means[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1)))
        return dec_out
    
    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        
        # TimesNetPro blocks
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        
        # Output
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None
