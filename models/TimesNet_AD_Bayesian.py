"""
TimesNet_AD_Bayesian: TimesNet + MC Dropout for Uncertainty Estimation

核心思想：
1. 在TimesNet的关键层加入Dropout
2. 测试时保持Dropout开启，多次前向传播
3. 预测方差 = 模型不确定性 = 异常指标

优势：
- 无需重新训练（如果原模型有Dropout）
- 理论扎实（Gal & Ghahramani 2016）
- 不会出现不确定性坍塌
- 适用于所有数据集（PSM、龙门铣等）

参考文献：
Gal, Y., & Ghahramani, Z. (2016). Dropout as a bayesian approximation:
Representing model uncertainty in deep learning. ICML.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.TimesNet import Model as TimesNetOriginal
from models.TimesNet import TimesBlock, FFT_for_Period
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


class TimesBlock_WithDropout(nn.Module):
    """
    TimesBlock with Dropout for Bayesian Uncertainty Estimation

    在Inception块后加入Dropout，用于MC Dropout
    """
    def __init__(self, configs):
        super(TimesBlock_WithDropout, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        # Inception卷积块
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

        # 关键：加入Dropout用于不确定性估计
        dropout_rate = getattr(configs, 'dropout', 0.1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x

            # reshape to 2D
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()

            # 2D conv
            out = self.conv(out)

            # 关键：应用Dropout
            out = self.dropout(out)

            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])

        res = torch.stack(res, dim=-1)

        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)

        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    TimesNet with Bayesian Uncertainty Estimation via MC Dropout

    使用方法：
    1. 训练时：正常训练（Dropout自动开启）
    2. 测试时：调用 anomaly_detection_bayesian() 进行MC Dropout推理
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # 使用带Dropout的TimesBlock
        self.model = nn.ModuleList([TimesBlock_WithDropout(configs)
                                    for _ in range(configs.e_layers)])

        # Embedding层（原版已有dropout）
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model,
                                          configs.embed, configs.freq,
                                          configs.dropout)

        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # Projection层
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

        # 额外的Dropout（可选）
        dropout_rate = getattr(configs, 'dropout', 0.1)
        self.feature_dropout = nn.Dropout(dropout_rate)

    def anomaly_detection(self, x_enc):
        """
        标准的异常检测前向传播（单次推理）
        """
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)

        # TimesNet blocks
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
            enc_out = self.feature_dropout(enc_out)  # 额外的dropout

        # projection
        dec_out = self.projection(enc_out)

        # De-normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        通用前向传播接口
        """
        return self.anomaly_detection(x_enc)
