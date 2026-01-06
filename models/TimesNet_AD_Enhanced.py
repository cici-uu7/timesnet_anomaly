"""
增强版 TimesNet_AD 模型
特性：
1. 多层架构：在每个TimesNet层后都添加AnomalyBlock
2. 动态Prior：混合静态高斯核和数据驱动Prior
3. 多层级特征融合：聚合所有层的异常信号
"""

import torch
import torch.nn as nn
from models.TimesNet import Model as TimesNetOriginal
from layers.Anomaly_Attention_Enhanced import AnomalyBlockEnhanced


class Model(nn.Module):
    """
    TimesNet_AD 增强版

    架构：
    - TimesNet Layer 1 -> AnomalyBlock 1 -> features_1
    - TimesNet Layer 2 -> AnomalyBlock 2 -> features_2
    - TimesNet Layer 3 -> AnomalyBlock 3 -> features_3
    - Multi-level Fusion -> Final Output
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.e_layers = configs.e_layers

        # 配置参数
        sigma_init_factor = getattr(configs, 'sigma_init_factor', 5.0)
        self.dynamic_prior = getattr(configs, 'dynamic_prior', False)  # 默认关闭动态Prior
        self.fusion_method = getattr(configs, 'fusion_method', 'weighted')  # 'weighted', 'attention', 'concat'

        # 1. 骨干网络
        self.timesnet = TimesNetOriginal(configs)

        # 2. 多层 AnomalyBlock（每层一个）
        self.anomaly_blocks = nn.ModuleList([
            AnomalyBlockEnhanced(
                d_model=configs.d_model,
                n_heads=configs.n_heads,
                win_size=configs.seq_len,
                d_ff=configs.d_ff,
                dropout=configs.dropout,
                sigma_init_factor=sigma_init_factor,
                dynamic_prior=self.dynamic_prior
            ) for _ in range(configs.e_layers)
        ])

        # 3. 多层级特征融合
        if self.fusion_method == 'weighted':
            # 可学习的权重
            self.layer_weights = nn.Parameter(torch.ones(configs.e_layers) / configs.e_layers)
        elif self.fusion_method == 'attention':
            # 注意力融合
            self.fusion_attention = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model // 2),
                nn.Tanh(),
                nn.Linear(configs.d_model // 2, 1)
            )
        elif self.fusion_method == 'concat':
            # 拼接后降维
            self.fusion_project = nn.Linear(configs.d_model * configs.e_layers, configs.d_model)

        # 4. 投影层
        self.projection = nn.Linear(configs.d_model, configs.c_out)

    def _fuse_multi_level_features(self, features_list):
        """
        融合多层级特征
        输入: features_list - List of [B, L, D] tensors
        输出: fused_features [B, L, D]
        """
        if self.fusion_method == 'weighted':
            # 加权融合
            weights = torch.softmax(self.layer_weights, dim=0)
            fused = sum(w * feat for w, feat in zip(weights, features_list))

        elif self.fusion_method == 'attention':
            # 注意力融合
            stacked = torch.stack(features_list, dim=1)  # [B, num_layers, L, D]
            B, num_layers, L, D = stacked.shape

            # 计算每层的注意力分数
            attn_scores = self.fusion_attention(stacked)  # [B, num_layers, L, 1]
            attn_weights = torch.softmax(attn_scores, dim=1)  # [B, num_layers, L, 1]

            # 加权求和
            fused = (stacked * attn_weights).sum(dim=1)  # [B, L, D]

        elif self.fusion_method == 'concat':
            # 拼接后降维
            concatenated = torch.cat(features_list, dim=-1)  # [B, L, D*num_layers]
            fused = self.fusion_project(concatenated)  # [B, L, D]

        return fused

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        前向传播

        返回:
            output: [B, L, C] - 重构输出
            all_series_attn: List of [B, H, L, L] - 每层的Series注意力
            all_prior_attn: List of [B, H, L, L] or [H, L, L] - 每层的Prior注意力
        """
        # 步骤 1: Embedding
        enc_out = self.timesnet.enc_embedding(x_enc, x_mark_enc)

        # 步骤 2: 多层 TimesNet + AnomalyBlock
        all_features = []
        all_series_attn = []
        all_prior_attn = []

        timesnet_feat = enc_out
        for i in range(self.timesnet.layer):
            # TimesNet 层
            timesnet_feat = self.timesnet.layer_norm(
                self.timesnet.model[i](timesnet_feat)
            )

            # AnomalyBlock 层
            enhanced_feat, series_attn, prior_attn = self.anomaly_blocks[i](timesnet_feat)

            # 收集多层级信息
            all_features.append(enhanced_feat)
            all_series_attn.append(series_attn)
            all_prior_attn.append(prior_attn)

            # 更新特征（用于下一层）
            timesnet_feat = enhanced_feat

        # 步骤 3: 多层级特征融合
        fused_features = self._fuse_multi_level_features(all_features)

        # 步骤 4: 最终投影
        output = self.projection(fused_features)

        # 返回值
        if self.output_attention:
            # 返回所有层的注意力（用于训练时的多层Loss）
            return output, all_series_attn, all_prior_attn
        else:
            # 只返回最后一层的注意力
            return output, all_series_attn[-1], all_prior_attn[-1]
