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

        # 2. AnomalyBlock 初始化 - 传入sigma初始化因子
        sigma_init_factor = getattr(configs, 'sigma_init_factor', 5.0)  # 默认5.0
        self.anomaly_block = AnomalyBlock(
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            win_size=configs.seq_len,
            d_ff=configs.d_ff,
            dropout=configs.dropout,
            sigma_init_factor=sigma_init_factor
        )

        # 3. 投影层
        self.projection = nn.Linear(configs.d_model, configs.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 步骤 1: Embedding
        enc_out = self.timesnet.enc_embedding(x_enc, x_mark_enc)

        # 步骤 2: TimesNet 特征提取（深层特征）
        # TimesNet 通过 2D 变换捕捉周期性特征
        timesnet_feat = enc_out
        for i in range(self.timesnet.layer):
            timesnet_feat = self.timesnet.layer_norm(self.timesnet.model[i](timesnet_feat))

        # 步骤 3: 传入 AnomalyBlock
        # 关键修改：只传深层特征，Q/K/V 和 Sigma 都从这个深层特征生成
        # 这样 Sigma 能根据上下文动态调整，而不是静态的
        out, series_attn, prior_attn = self.anomaly_block(timesnet_feat)

        # 步骤 4: 最终投影
        output = self.projection(out)

        return output, series_attn, prior_attn