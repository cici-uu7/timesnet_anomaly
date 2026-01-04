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

        # 1. 骨干网络
        self.timesnet = TimesNetOriginal(configs)

        # 2. AnomalyBlock 初始化
        # 关键修改：输入维度是 d_model (不再是 d_model * 2)
        self.anomaly_block = AnomalyBlock(
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            win_size=configs.seq_len,
            d_ff=configs.d_ff,
            dropout=configs.dropout
        )

        # 3. 投影层
        # 关键修改：输入维度是 d_model
        self.projection = nn.Linear(configs.d_model, configs.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 步骤 1: Embedding (作为 Prior Branch 的输入)
        # 保留原始的局部特征
        enc_out = self.timesnet.enc_embedding(x_enc, x_mark_enc)
        raw_embedding = enc_out.clone()

        # 步骤 2: TimesNet 特征提取 (作为 Series Branch 的输入)
        # 周期性提取后包含了深层规律
        timesnet_feat = enc_out
        for i in range(self.timesnet.layer):
            timesnet_feat = self.timesnet.layer_norm(self.timesnet.model[i](timesnet_feat))

        # 步骤 3: 分别传入 AnomalyBlock
        # 方案二核心：深层给Series，浅层给Prior
        out, series_attn, prior_attn = self.anomaly_block(x_series= timesnet_feat, x_prior = raw_embedding)

        # 步骤 4: 最终投影
        output = self.projection(out)

        return output, series_attn, prior_attn