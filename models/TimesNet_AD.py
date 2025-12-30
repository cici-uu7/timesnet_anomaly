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

        # 1. 实例化原版 TimesNet 作为骨干网络
        self.timesnet = TimesNetOriginal(configs)

        # 2. 定义关联差异模块
        # 输入是 [TimesNet深层特征; 原始Embedding]，所以维度是 d_model * 2
        self.fusion_dim = configs.d_model * 2
        self.anomaly_block = AnomalyBlock(
            d_model=self.fusion_dim,
            n_heads=configs.n_heads,
            win_size=configs.seq_len,
            d_ff=configs.d_ff,
            dropout=configs.dropout
        )

        # 3. 投影层：映射回输出维度
        self.projection = nn.Linear(self.fusion_dim, configs.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 步骤 1: Embedding (直接复用 TimesNet 的 Embedding 层)
        # enc_out: [Batch, Seq_Len, d_model]
        enc_out = self.timesnet.enc_embedding(x_enc, x_mark_enc)

        # 备份“浅层特征”用于后续拼接
        raw_embedding = enc_out.clone()

        # 步骤 2: TimesNet 特征提取 (手动遍历层，替代不存在的 .encoder)
        # 参考 TimesNet.py 中的 anomaly_detection 方法逻辑
        timesnet_feat = enc_out
        for i in range(self.timesnet.layer):
            # 通过 TimesBlock 提取 2D 时序变化特征
            timesnet_feat = self.timesnet.layer_norm(self.timesnet.model[i](timesnet_feat))

        # 此时 timesnet_feat 是“深层特征”，维度 [Batch, Seq_Len, d_model]

        # 步骤 3: 特征融合
        # 拼接深层特征和浅层特征
        combined_feat = torch.cat([timesnet_feat, raw_embedding], dim=-1)

        # 步骤 4: 关联差异计算 (Anomaly Attention)
        # out: [Batch, Seq_Len, 2*d_model]
        out, series_attn, prior_attn = self.anomaly_block(combined_feat)

        # 步骤 5: 重构输出
        output = self.projection(out)

        return output, series_attn, prior_attn