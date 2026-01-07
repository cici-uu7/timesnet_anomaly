"""
TimesNet_AD_VAD: TimesNet + Variable Association Discrepancy

核心思想：
1. TimesNet 负责时序重构（完全不变）
2. 变量关联差异模块检测变量间关系异常（辅助机制）
3. 异常分数 = α * 重构误差 + β * 变量关联差异

变量关联差异（Variable Association Discrepancy）：
- Prior: 物理约束 + 统计相关性的融合（固定）
- Series: 当前窗口的变量相关性（动态）
- Discrepancy: KL(Prior, Series) - 偏离正常关联模式的程度

Prior 融合策略：
    hybrid_prior = λ * physical_prior + (1-λ) * statistical_prior

- λ = 1.0：完全信任物理约束
- λ = 0.5：物理和统计各占一半（推荐）
- λ = 0.0：完全用统计（无物理知识时）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.TimesNet import Model as TimesNetOriginal


def build_physical_prior(num_vars, var_groups=None, cross_group_relations=None):
    """
    构建物理约束 Prior

    Args:
        num_vars: 变量数量
        var_groups: 变量分组，例如 {
            'temperature': [0, 1, 2],      # 温度相关变量的索引
            'vibration': [3, 4, 5],        # 振动相关变量
            'current': [6, 7],             # 电流相关变量
            'speed': [8, 9]                # 速度相关变量
        }
        cross_group_relations: 跨组关联，例如 {
            ('temperature', 'vibration'): 0.6,  # 温度和振动有中等关联
            ('current', 'speed'): 0.8,          # 电流和速度有强关联
        }

    Returns:
        prior: [V, V] 物理约束矩阵
    """
    prior = torch.zeros(num_vars, num_vars)

    if var_groups is None:
        # 默认：均匀分布（无物理约束）
        prior = torch.ones(num_vars, num_vars) / num_vars
        return prior

    # 1. 同组内强关联
    for group_name, indices in var_groups.items():
        for i in indices:
            for j in indices:
                if i < num_vars and j < num_vars:
                    prior[i, j] = 1.0  # 同组强关联

    # 2. 跨组关联
    if cross_group_relations:
        for (g1, g2), strength in cross_group_relations.items():
            if g1 in var_groups and g2 in var_groups:
                for i in var_groups[g1]:
                    for j in var_groups[g2]:
                        if i < num_vars and j < num_vars:
                            prior[i, j] = strength
                            prior[j, i] = strength

    # 3. 对角线（自相关）
    for i in range(num_vars):
        prior[i, i] = 1.0

    # 4. 未指定的变量对：弱关联
    prior[prior == 0] = 0.1

    return prior


def build_distance_prior(num_vars, sigma=3.0):
    """
    构建基于变量索引距离的 Prior（简化版物理约束）

    假设：索引相近的变量可能是同类型传感器，应有更强关联

    Args:
        num_vars: 变量数量
        sigma: 高斯核宽度，控制关联衰减速度

    Returns:
        prior: [V, V] 距离 Prior 矩阵
    """
    prior = torch.zeros(num_vars, num_vars)

    for i in range(num_vars):
        for j in range(num_vars):
            distance = abs(i - j)
            # 高斯衰减：距离越远，关联越弱
            prior[i, j] = torch.exp(torch.tensor(-distance ** 2 / (2 * sigma ** 2)))

    return prior


class VariableAssociationModule(nn.Module):
    """
    变量关联差异模块

    计算变量间的实时相关性（Series），与预设的 Prior 比较

    Prior 融合: hybrid_prior = λ * physical + (1-λ) * statistical
    """

    def __init__(self, num_vars, d_model, n_heads=4):
        """
        Args:
            num_vars: 变量数量
            d_model: 模型维度
            n_heads: 注意力头数（暂未使用）
        """
        super().__init__()
        self.num_vars = num_vars
        self.d_model = d_model
        self.n_heads = n_heads

        # Prior 将在外部设置
        self.register_buffer('prior', None)

    def set_prior(self, prior_matrix):
        """
        设置 Prior 矩阵

        Args:
            prior_matrix: [V, V] 变量间的期望关联（已经是平方归一化后的概率分布）

        注意: Prior 应该已经和 Series 使用相同的归一化方式（平方归一化）
              不要再用 softmax，否则会把对角线值（自相关）拉低
        """
        # 确保是概率分布（每行和为1）
        # 不用softmax，因为softmax会破坏对角线的高值
        prior = prior_matrix / (prior_matrix.sum(dim=-1, keepdim=True) + 1e-8)

        # 需要重新注册 buffer
        if 'prior' in self._buffers:
            del self._buffers['prior']
        self.register_buffer('prior', prior)

    def compute_variable_correlation(self, x):
        """
        从原始输入计算变量间相关性（Series）

        Args:
            x: [B, L, V] 原始输入
        Returns:
            series: [B, V, V] 变量间相关性矩阵（概率分布）
        """
        B, L, V = x.shape

        # 标准化
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True) + 1e-8
        x_norm = (x - x_mean) / x_std

        # 计算相关性矩阵 (Pearson correlation)
        # corr[i,j] ≈ correlation between var_i and var_j
        # corr[i,i] ≈ 1.0 (self-correlation)
        corr = torch.bmm(x_norm.transpose(1, 2), x_norm) / L  # [B, V, V]

        # 修复: 不用 softmax (会把对角线拉平到 1/V)
        # 改用平方归一化: 放大差异,同时保持对角线突出
        series = corr ** 2  # [0, 1], 强相关→接近1, 弱相关→接近0
        series = series / (series.sum(dim=-1, keepdim=True) + 1e-8)  # 归一化为概率分布

        return series

    def forward(self, x, features=None):
        """
        计算变量关联差异

        Args:
            x: [B, L, V] 原始输入
            features: [B, L, D] TimesNet 隐藏特征（暂不使用）
        Returns:
            discrepancy: [B] 每个样本的变量关联差异
            series: [B, V, V] 实时变量相关性
        """
        if self.prior is None:
            B = x.shape[0]
            return torch.zeros(B, device=x.device), None

        # 计算 Series（实时变量相关性）
        series = self.compute_variable_correlation(x)

        # 扩展 Prior 到 batch 维度
        prior = self.prior.unsqueeze(0).expand(x.shape[0], -1, -1)

        # 双向 KL 散度
        kl_ps = prior * (torch.log(prior + 1e-8) - torch.log(series + 1e-8))
        kl_sp = series * (torch.log(series + 1e-8) - torch.log(prior + 1e-8))

        # 总差异
        discrepancy = (kl_ps.sum(dim=-1).sum(dim=-1) + kl_sp.sum(dim=-1).sum(dim=-1)) / 2

        return discrepancy, series


class Model(nn.Module):
    """
    TimesNet_AD_VAD: TimesNet + Variable Association Discrepancy

    架构：
    1. TimesNet 骨干网络（重构）
    2. 变量关联差异模块（辅助检测）

    训练：只优化重构损失（TimesNet 不变）
    测试：融合重构误差 + 变量关联差异

    Prior 融合: hybrid_prior = λ * physical + (1-λ) * statistical
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_vars = configs.enc_in

        # 1. TimesNet 骨干（完全复用原版）
        self.timesnet = TimesNetOriginal(configs)

        # 2. 变量关联差异模块
        self.vad_module = VariableAssociationModule(
            num_vars=configs.enc_in,
            d_model=configs.d_model,
            n_heads=getattr(configs, 'n_heads', 4)
        )

    def set_prior(self, prior_matrix):
        """设置变量关联 Prior"""
        self.vad_module.set_prior(prior_matrix)

    def set_physical_prior(self, var_groups, cross_group_relations=None):
        """
        设置物理约束 Prior

        Args:
            var_groups: 变量分组字典
            cross_group_relations: 跨组关联字典
        """
        prior = build_physical_prior(
            self.num_vars,
            var_groups,
            cross_group_relations
        )
        self.set_prior(prior)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        前向传播

        Args:
            x_enc: [B, L, V] 输入序列
        Returns:
            output: [B, L, V] 重构输出
            var_discrepancy: [B] 变量关联差异
            series: [B, V, V] 实时变量相关性
        """
        # 1. TimesNet 重构（完全不变）
        output = self.timesnet.anomaly_detection(x_enc)

        # 2. 计算变量关联差异
        var_discrepancy, series = self.vad_module(x_enc)

        return output, var_discrepancy, series
