import torch
import torch.nn as nn
from models.TimesNet import Model as TimesNetOriginal
from layers.Anomaly_Attention import AnomalyBlock


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # 1. 实例化原版 TimesNet
        self.timesnet = TimesNetOriginal(configs)

        # 2. 定义关联差异模块
        # 我们使用 Concat(TimesNet特征, 原始Embedding)，所以输入维度是 d_model * 2
        self.fusion_dim = configs.d_model * 2
        self.anomaly_block = AnomalyBlock(
            d_model=self.fusion_dim,
            n_heads=configs.n_heads,
            win_size=configs.seq_len,
            d_ff=configs.d_ff,
            dropout=configs.dropout
        )

        # 3. 投影层：把融合后的特征映射回输出维度 (c_out)
        self.projection = nn.Linear(self.fusion_dim, configs.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 步骤 A: 利用 TimesNet 提取深层特征
        # enc_out: [B, L, d_model]
        enc_out = self.timesnet.enc_embedding(x_enc, x_mark_enc)
        raw_embedding = enc_out.clone()  # 备份原始 Embedding

        # timesnet_feat: [B, L, d_model]
        # 注意：TimesNet 的 encoder 可能返回 (x, attns) 元组，我们只需要 x
        timesnet_out = self.timesnet.encoder(enc_out, x_mark_enc)
        if isinstance(timesnet_out, tuple):
            timesnet_feat = timesnet_out[0]
        else:
            timesnet_feat = timesnet_out

        # 步骤 B: 特征融合 (Concatenation)
        # 将 "深层周期特征" 与 "浅层局部特征" 拼接，防止过平滑
        combined_feat = torch.cat([timesnet_feat, raw_embedding], dim=-1)

        # 步骤 C: 关联差异计算
        # out: [B, L, 2*d_model]
        # series_attn, prior_attn: [B, n_heads, L, L]
        out, series_attn, prior_attn = self.anomaly_block(combined_feat)

        # 步骤 D: 重构输出
        output = self.projection(out)

        # 训练和测试时都需要 Attention Map 来计算差异损失
        return output, series_attn, prior_attn