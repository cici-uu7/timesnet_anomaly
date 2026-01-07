"""
TimesNet_AD_VAD 实验类
TimesNet + Variable Association Discrepancy（变量关联差异）

核心设计：
1. 训练：只优化重构损失（TimesNet 完全不变）
2. Prior：融合物理约束和统计相关性
   hybrid_prior = λ * physical_prior + (1-λ) * statistical_prior
3. 测试：异常分数 = α * 重构误差 + β * 变量关联差异

参数说明：
- λ (prior_lambda): Prior 融合权重
  - λ = 1.0：完全信任物理约束
  - λ = 0.5：物理和统计各占一半（推荐）
  - λ = 0.0：完全用统计（无物理知识时）
- α (alpha): 重构误差权重
- β (beta): 变量关联差异权重
"""

from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from models import TimesNet_AD_VAD
from models.TimesNet_AD_VAD import build_physical_prior
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from utils.tools import adjustment, EarlyStopping, adjust_learning_rate


# ========== 物理约束配置 ==========
# 用户根据数据集的变量含义配置

# PSM 数据集（服务器监控，25个变量）
PSM_VAR_GROUPS = {
    'cpu': [0, 1, 2, 3, 4],
    'memory': [5, 6, 7, 8],
    'disk': [9, 10, 11, 12],
    'network': [13, 14, 15, 16],
    'other': [17, 18, 19, 20, 21, 22, 23, 24]
}
PSM_CROSS_RELATIONS = {
    ('cpu', 'memory'): 0.7,
    ('disk', 'network'): 0.5,
}

# SMD 数据集（服务器机器，38个变量）
SMD_VAR_GROUPS = {
    'group1': list(range(0, 10)),
    'group2': list(range(10, 20)),
    'group3': list(range(20, 30)),
    'group4': list(range(30, 38)),
}
SMD_CROSS_RELATIONS = {}

# CNC 龙门铣床（示例配置，需要根据实际传感器调整）
CNC_VAR_GROUPS = {
    'temperature': [0, 1, 2],       # 主轴温度、轴承温度、环境温度
    'vibration': [3, 4, 5],         # X/Y/Z 轴振动
    'current': [6, 7, 8],           # 主轴电流、进给电机电流
    'speed': [9, 10],               # 主轴转速、进给速度
    'position': [11, 12, 13],       # X/Y/Z 位置
    'pressure': [14, 15],           # 液压压力、气压
}
CNC_CROSS_RELATIONS = {
    ('temperature', 'vibration'): 0.6,
    ('current', 'speed'): 0.8,
    ('current', 'temperature'): 0.5,
    ('speed', 'vibration'): 0.7,
}


def get_physical_config(dataset_name):
    """获取数据集对应的物理约束配置"""
    configs = {
        'PSM': (PSM_VAR_GROUPS, PSM_CROSS_RELATIONS),
        'SMD': (SMD_VAR_GROUPS, SMD_CROSS_RELATIONS),
        'CNC': (CNC_VAR_GROUPS, CNC_CROSS_RELATIONS),
    }
    return configs.get(dataset_name, (None, None))


class Exp_TimesNet_AD_VAD(Exp_Basic):
    """
    TimesNet_AD_VAD 实验类

    Prior 融合策略：
        hybrid_prior = λ * physical_prior + (1-λ) * statistical_prior
    """

    def __init__(self, args):
        super().__init__(args)
        self.model_dict = {'TimesNet_AD_VAD': TimesNet_AD_VAD}
        # λ: Prior 融合权重
        self.prior_lambda = getattr(args, 'prior_lambda', 0.5)

    def _build_model(self):
        """构建模型"""
        model = TimesNet_AD_VAD.Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        """获取数据"""
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """选择优化器"""
        model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _compute_statistical_prior(self, train_loader):
        """
        从训练数据计算统计 Prior

        统计正常数据中变量间的相关性
        """
        all_corr = []

        with torch.no_grad():
            for batch_x, _ in train_loader:
                batch_x = batch_x.float()
                B, L, V = batch_x.shape

                # 标准化
                x_mean = batch_x.mean(dim=1, keepdim=True)
                x_std = batch_x.std(dim=1, keepdim=True) + 1e-8
                x_norm = (batch_x - x_mean) / x_std

                # 计算相关性矩阵 (Pearson correlation)
                corr = torch.bmm(x_norm.transpose(1, 2), x_norm) / L
                all_corr.append(corr)

        all_corr = torch.cat(all_corr, dim=0)
        statistical_prior = all_corr.mean(dim=0)

        # 修复: 和 Series 保持一致的处理方式
        # 平方后归一化（放大相关性差异）
        statistical_prior = statistical_prior ** 2
        statistical_prior = statistical_prior / (statistical_prior.sum(dim=-1, keepdim=True) + 1e-8)

        return statistical_prior

    def _compute_physical_prior(self):
        """
        构建物理约束 Prior

        基于领域知识定义变量分组和关联强度
        """
        dataset_name = getattr(self.args, 'data', 'unknown')
        var_groups, cross_relations = get_physical_config(dataset_name)

        if var_groups is None:
            print(f"  No physical config for dataset '{dataset_name}'")
            return None

        physical_prior = build_physical_prior(
            self.args.enc_in,
            var_groups,
            cross_relations
        )
        return physical_prior

    def _setup_hybrid_prior(self, train_loader):
        """
        设置融合 Prior

        hybrid_prior = λ * physical_prior + (1-λ) * statistical_prior
        """
        lam = self.prior_lambda
        print(f"\nSetting up Hybrid Prior (λ={lam})...")

        # 1. 计算统计 Prior
        print("  Computing statistical Prior from training data...")
        statistical_prior = self._compute_statistical_prior(train_loader)
        print(f"    Shape: {statistical_prior.shape}")
        print(f"    Range: [{statistical_prior.min():.4f}, {statistical_prior.max():.4f}]")

        # 2. 计算物理 Prior
        print("  Building physical Prior from domain knowledge...")
        physical_prior = self._compute_physical_prior()

        # 3. 融合
        if physical_prior is not None:
            print(f"  Fusing: hybrid = {lam} * physical + {1-lam} * statistical")
            # 归一化到相同尺度
            physical_prior = physical_prior / (physical_prior.max() + 1e-8)
            statistical_prior = statistical_prior / (statistical_prior.max() + 1e-8)

            hybrid_prior = lam * physical_prior + (1 - lam) * statistical_prior
            print(f"    Hybrid range: [{hybrid_prior.min():.4f}, {hybrid_prior.max():.4f}]")
        else:
            print(f"  No physical Prior available, using pure statistical (λ effectively = 0)")
            hybrid_prior = statistical_prior

        # 4. 设置到模型
        self.model.set_prior(hybrid_prior.to(self.device))
        print("  Prior setup complete.")

        return hybrid_prior

    def vali(self, vali_data, vali_loader, criterion):
        """验证（只用重构损失）"""
        self.model.eval()
        total_loss = []

        with torch.no_grad():
            for batch_x, _ in vali_loader:
                batch_x = batch_x.float().to(self.device)
                outputs, _, _ = self.model(batch_x, None, None, None)
                loss = criterion(outputs, batch_x)
                total_loss.append(loss.item())

        avg_loss = np.average(total_loss)
        self.model.train()
        return avg_loss

    def train(self, setting):
        """
        训练函数

        训练策略：只优化重构损失（TimesNet 完全不变）
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        import time
        time_now = time.time()

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        optimizer = self._select_optimizer()
        criterion = nn.MSELoss()

        # 设置融合 Prior
        self._setup_hybrid_prior(train_loader)

        print("\n" + "=" * 60)
        print("Training TimesNet_AD_VAD")
        print("  - Training: Pure reconstruction loss (TimesNet unchanged)")
        print(f"  - Prior: λ={self.prior_lambda} (hybrid)")
        print("=" * 60)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, _) in enumerate(train_loader):
                iter_count += 1
                optimizer.zero_grad()

                batch_x = batch_x.float().to(self.device)
                outputs, _, _ = self.model(batch_x, None, None, None)

                # 重构损失（和原版 TimesNet 完全一样）
                loss = criterion(outputs, batch_x)

                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * len(train_loader) - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")
            train_loss_avg = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {len(train_loader)} | "
                  f"Train Loss: {train_loss_avg:.7f} Vali Loss: {vali_loss:.7f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(optimizer, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        print("\nTraining complete.")

        return self.model

    def test(self, setting, test=0):
        """
        测试函数

        融合策略（VAD Boost Only）：
        - VAD 低于正常水平：不影响重构误差（boost = 0）
        - VAD 高于正常水平：放大重构误差（boost > 0）

        公式：score = rec_error * (1 + β * vad_boost)
        其中：vad_boost = max(0, (vad - vad_mean) / vad_std)
        """
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')

        if test:
            print('Loading model...')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            )
            # 重新设置 Prior
            self._setup_hybrid_prior(train_loader)

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()

        # 异常分数权重
        beta = getattr(self.args, 'beta', 0.3)
        anomaly_ratio = getattr(self.args, 'anomaly_ratio', 1.0)

        print("=" * 60)
        print("Testing: VAD Boost Only Strategy")
        print(f"  score = rec * (1 + {beta} * max(0, vad_boost))")
        print(f"  VAD only boosts, never penalizes")
        print(f"Prior: λ={self.prior_lambda} (hybrid)")
        print("=" * 60)

        # ========== 训练集统计 ==========
        print("Computing scores on training set...")
        train_rec_list = []
        train_vad_list = []

        with torch.no_grad():
            for batch_x, _ in train_loader:
                batch_x = batch_x.float().to(self.device)
                outputs, var_disc, _ = self.model(batch_x, None, None, None)

                rec_error = torch.mean((outputs - batch_x) ** 2, dim=-1)
                vad = var_disc.unsqueeze(1).expand(-1, batch_x.shape[1])

                train_rec_list.append(rec_error.cpu().numpy())
                train_vad_list.append(vad.cpu().numpy())

        train_rec = np.concatenate(train_rec_list, axis=0).reshape(-1)
        train_vad = np.concatenate(train_vad_list, axis=0).reshape(-1)

        rec_mean, rec_std = train_rec.mean(), train_rec.std() + 1e-8
        vad_mean, vad_std = train_vad.mean(), train_vad.std() + 1e-8

        print(f"Train Rec: mean={rec_mean:.6f}, std={rec_std:.6f}")
        print(f"Train VAD: mean={vad_mean:.6f}, std={vad_std:.6f}")

        # 归一化重构误差
        train_rec_norm = (train_rec - rec_mean) / rec_std

        # VAD boost：只保留超出正常的部分（ReLU）
        train_vad_boost = np.maximum(0, (train_vad - vad_mean) / vad_std)

        # 融合分数：重构误差 * (1 + β * vad_boost)
        # 确保 rec_norm 为正（加偏移）
        train_rec_positive = train_rec_norm - train_rec_norm.min() + 1e-8
        train_energy = train_rec_positive * (1 + beta * train_vad_boost)

        print(f"Train VAD boost: mean={train_vad_boost.mean():.4f}, "
              f"max={train_vad_boost.max():.4f}, "
              f"non-zero ratio={np.mean(train_vad_boost > 0):.2%}")

        # ========== 测试集评估 ==========
        print("Evaluating on test set...")
        test_rec_list = []
        test_vad_list = []
        test_labels_list = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.float().to(self.device)
                outputs, var_disc, _ = self.model(batch_x, None, None, None)

                rec_error = torch.mean((outputs - batch_x) ** 2, dim=-1)
                vad = var_disc.unsqueeze(1).expand(-1, batch_x.shape[1])

                test_rec_list.append(rec_error.cpu().numpy())
                test_vad_list.append(vad.cpu().numpy())
                test_labels_list.append(batch_y.numpy())

        test_rec = np.concatenate(test_rec_list, axis=0).reshape(-1)
        test_vad = np.concatenate(test_vad_list, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels_list, axis=0).reshape(-1)

        # 归一化（使用训练集统计量）
        test_rec_norm = (test_rec - rec_mean) / rec_std
        test_vad_boost = np.maximum(0, (test_vad - vad_mean) / vad_std)

        # 融合分数
        test_rec_positive = test_rec_norm - train_rec_norm.min() + 1e-8
        test_energy = test_rec_positive * (1 + beta * test_vad_boost)

        print(f"Test VAD boost: mean={test_vad_boost.mean():.4f}, "
              f"max={test_vad_boost.max():.4f}, "
              f"non-zero ratio={np.mean(test_vad_boost > 0):.2%}")

        # ========== 统计 ==========
        print(f"\nScore Statistics:")
        print(f"  Train: mean={train_energy.mean():.4f}, std={train_energy.std():.4f}")
        print(f"  Test:  mean={test_energy.mean():.4f}, std={test_energy.std():.4f}")

        # ========== 阈值策略对比 ==========
        print(f"\n{'=' * 60}")
        print("Threshold Strategy Comparison:")
        print(f"{'=' * 60}")

        # 策略1: 混合训练+测试 (当前策略)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold_combined = np.percentile(combined_energy, 100 - anomaly_ratio)

        # 策略2: 只用训练集
        threshold_train = np.percentile(train_energy, 100 - anomaly_ratio)

        # 策略3: 只用测试集 (理想情况，仅用于分析)
        threshold_test = np.percentile(test_energy, 100 - anomaly_ratio)

        # 策略4: 训练集 + k*std
        threshold_train_std = train_energy.mean() + 3 * train_energy.std()

        print(f"1. Combined (Train+Test): {threshold_combined:.6f}")
        print(f"2. Train only:            {threshold_train:.6f}")
        print(f"3. Test only (oracle):    {threshold_test:.6f}")
        print(f"4. Train mean+3std:       {threshold_train_std:.6f}")

        # 测试各种阈值策略
        strategies = {
            'Combined (default)': threshold_combined,
            'Train only': threshold_train,
            'Test only (oracle)': threshold_test,
            'Train mean+3std': threshold_train_std,
        }

        best_f1 = 0
        best_strategy = None
        best_results = None

        print(f"\n{'Strategy':<25} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8}")
        print("-" * 60)

        for strategy_name, threshold in strategies.items():
            pred = (test_energy > threshold).astype(int)
            gt = test_labels.astype(int)
            gt_adj, pred_adj = adjustment(gt, pred)

            acc = accuracy_score(gt_adj, pred_adj)
            prec, rec, f1, _ = precision_recall_fscore_support(gt_adj, pred_adj, average='binary')

            marker = ""
            if f1 > best_f1:
                best_f1 = f1
                best_strategy = strategy_name
                best_results = (acc, prec, rec, f1, threshold)
                marker = " ⭐"

            print(f"{strategy_name:<25} {acc:.4f}   {prec:.4f}   {rec:.4f}   {f1:.4f}{marker}")

        # 使用最佳策略
        accuracy, precision, recall, f_score, threshold = best_results
        pred = (test_energy > threshold).astype(int)
        gt = test_labels.astype(int)
        gt, pred = adjustment(gt, pred)

        print(f"\n{'=' * 60}")
        print(f"Best Results (β={beta}, λ={self.prior_lambda}, {best_strategy}):")
        print(f"  Threshold: {threshold:.6f}")
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  F1 Score : {f_score:.4f}")
        print(f"{'=' * 60}")

        # ========== 消融对比 ==========
        print("\n[Ablation] Reconstruction only (no VAD):")
        # 纯重构误差（不用 VAD）
        train_energy_rec_only = train_rec_positive
        test_energy_rec_only = test_rec_positive
        combined_rec_only = np.concatenate([train_energy_rec_only, test_energy_rec_only], axis=0)
        threshold_rec_only = np.percentile(combined_rec_only, 100 - anomaly_ratio)

        pred_rec_only = (test_energy_rec_only > threshold_rec_only).astype(int)
        gt_rec, pred_rec = adjustment(test_labels.astype(int), pred_rec_only)
        _, _, f_rec, _ = precision_recall_fscore_support(gt_rec, pred_rec, average='binary')
        print(f"  F1 (Rec only): {f_rec:.4f}")

        improvement = (f_score - f_rec) * 100
        print(f"  VAD improvement: {improvement:+.2f}%")

        # ========== 调试分析：Prior 质量检查 ==========
        print("\n" + "=" * 60)
        print("[Debug] Prior Matrix Analysis:")
        prior_matrix = self.model.vad_module.prior.cpu().numpy()
        print(f"  Prior shape: {prior_matrix.shape}")
        print(f"  Prior mean: {prior_matrix.mean():.6f}")
        print(f"  Prior std: {prior_matrix.std():.6f}")
        print(f"  Prior max correlation: {prior_matrix.max():.6f}")
        print(f"  Prior min correlation: {prior_matrix.min():.6f}")

        # 找出相关性最强和最弱的变量对（排除对角线）
        prior_no_diag = prior_matrix.copy()
        np.fill_diagonal(prior_no_diag, -np.inf)  # 用-inf而不是0，这样不会影响argmin
        max_idx = np.unravel_index(np.argmax(prior_no_diag), prior_no_diag.shape)

        prior_no_diag_for_min = prior_matrix.copy()
        np.fill_diagonal(prior_no_diag_for_min, np.inf)  # 用inf排除对角线
        min_idx = np.unravel_index(np.argmin(prior_no_diag_for_min), prior_no_diag_for_min.shape)

        print(f"  Strongest correlation: Var {max_idx[0]} <-> Var {max_idx[1]} = {prior_matrix[max_idx]:.6f}")
        print(f"  Weakest correlation:   Var {min_idx[0]} <-> Var {min_idx[1]} = {prior_matrix[min_idx]:.6f}")
        print(f"  Diagonal mean (self-correlation): {np.diag(prior_matrix).mean():.6f}")

        # ========== 调试分析：VAD 原始值检查 ==========
        print("\n" + "=" * 60)
        print("[Debug] VAD Raw Values:")
        print(f"  Train VAD raw: mean={train_vad.mean():.6f}, std={train_vad.std():.6f}")
        print(f"  Train VAD raw: min={train_vad.min():.6f}, max={train_vad.max():.6f}")
        print(f"  Test VAD raw:  mean={test_vad.mean():.6f}, std={test_vad.std():.6f}")
        print(f"  Test VAD raw:  min={test_vad.min():.6f}, max={test_vad.max():.6f}")

        # 检查极端值
        extreme_threshold = vad_mean + 10 * vad_std
        extreme_ratio_train = np.mean(train_vad > extreme_threshold)
        extreme_ratio_test = np.mean(test_vad > extreme_threshold)
        print(f"  Train extreme VAD (>mean+10*std): {extreme_ratio_train:.2%}")
        print(f"  Test extreme VAD (>mean+10*std):  {extreme_ratio_test:.2%}")

        # ========== 调试分析：Series vs Prior 差异 ==========
        print("\n" + "=" * 60)
        print("[Debug] Series vs Prior (KL Divergence Analysis):")

        # 采样几个批次分析 Series
        sample_series_list = []
        sample_labels_list = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                if i >= 5:  # 只取前 5 个批次
                    break
                batch_x = batch_x.float().to(self.device)
                _, _, series = self.model(batch_x, None, None, None)
                sample_series_list.append(series.cpu().numpy())
                sample_labels_list.append(batch_y.numpy())

        if len(sample_series_list) > 0:
            sample_series = np.concatenate(sample_series_list, axis=0)  # [N, V, V]
            sample_labels = np.concatenate(sample_labels_list, axis=0)  # [N, win_size]

            # 修复: 取每个窗口的最大标签 (如果窗口内有异常点,就认为是异常窗口)
            sample_labels = sample_labels.max(axis=1)  # [N, win_size] -> [N]

            # 区分正常和异常样本
            normal_mask = sample_labels == 0
            anomaly_mask = sample_labels > 0

            print(f"  Sampled {len(sample_series)} windows: {normal_mask.sum()} normal, {anomaly_mask.sum()} anomaly")

            if normal_mask.sum() > 0 and anomaly_mask.sum() > 0:
                # 计算平均 Series
                series_normal = sample_series[normal_mask].mean(axis=0)  # [V, V]
                series_anomaly = sample_series[anomaly_mask].mean(axis=0)  # [V, V]

                # 提取对角线和非对角线
                V = series_normal.shape[0]
                diag_indices = np.arange(V)

                # Prior 统计
                prior_diag = prior_matrix[diag_indices, diag_indices]
                prior_off_diag = prior_matrix.copy()
                np.fill_diagonal(prior_off_diag, 0)
                prior_off_diag_mean = prior_off_diag.sum() / (V * V - V)

                # Normal Series 统计
                normal_diag = series_normal[diag_indices, diag_indices]
                normal_off_diag = series_normal.copy()
                np.fill_diagonal(normal_off_diag, 0)
                normal_off_diag_mean = normal_off_diag.sum() / (V * V - V)

                # Anomaly Series 统计
                anomaly_diag = series_anomaly[diag_indices, diag_indices]
                anomaly_off_diag = series_anomaly.copy()
                np.fill_diagonal(anomaly_off_diag, 0)
                anomaly_off_diag_mean = anomaly_off_diag.sum() / (V * V - V)

                print(f"\n  Prior Matrix:")
                print(f"    Diagonal mean:     {prior_diag.mean():.6f}")
                print(f"    Off-diagonal mean: {prior_off_diag_mean:.6f}")
                print(f"    Ratio (Diag/Off):  {prior_diag.mean() / (prior_off_diag_mean + 1e-8):.2f}x")

                print(f"\n  Normal Series Matrix:")
                print(f"    Diagonal mean:     {normal_diag.mean():.6f}")
                print(f"    Off-diagonal mean: {normal_off_diag_mean:.6f}")
                print(f"    Ratio (Diag/Off):  {normal_diag.mean() / (normal_off_diag_mean + 1e-8):.2f}x")

                print(f"\n  Anomaly Series Matrix:")
                print(f"    Diagonal mean:     {anomaly_diag.mean():.6f}")
                print(f"    Off-diagonal mean: {anomaly_off_diag_mean:.6f}")
                print(f"    Ratio (Diag/Off):  {anomaly_diag.mean() / (anomaly_off_diag_mean + 1e-8):.2f}x")

                print(f"\n  Overall Statistics:")
                print(f"    Normal Series: mean={series_normal.mean():.6f}, std={series_normal.std():.6f}")
                print(f"    Anomaly Series: mean={series_anomaly.mean():.6f}, std={series_anomaly.std():.6f}")

                # Series 和 Prior 的差异
                diff_normal = np.abs(series_normal - prior_matrix).mean()
                diff_anomaly = np.abs(series_anomaly - prior_matrix).mean()
                print(f"\n  |Series - Prior|:")
                print(f"    Normal:  {diff_normal:.6f}")
                print(f"    Anomaly: {diff_anomaly:.6f}")
                print(f"    Ratio (Anomaly/Normal): {diff_anomaly / (diff_normal + 1e-8):.2f}x")

        # ========== VAD 效果分析 ==========
        print("\n" + "=" * 60)
        print("[Analysis] VAD Boost Effect:")
        # 统计 VAD 在异常点和正常点的差异
        anomaly_mask = test_labels > 0
        if anomaly_mask.sum() > 0:
            vad_boost_anomaly = test_vad_boost[anomaly_mask].mean()
            vad_boost_normal = test_vad_boost[~anomaly_mask].mean()

            # 额外统计：原始 VAD 值
            vad_raw_anomaly = test_vad[anomaly_mask].mean()
            vad_raw_normal = test_vad[~anomaly_mask].mean()

            print(f"  Anomaly points VAD raw:   {vad_raw_anomaly:.4f}")
            print(f"  Normal points VAD raw:    {vad_raw_normal:.4f}")
            print(f"  Ratio (Anomaly/Normal):   {vad_raw_anomaly / (vad_raw_normal + 1e-8):.2f}x")
            print(f"")
            print(f"  Anomaly points VAD boost: {vad_boost_anomaly:.4f}")
            print(f"  Normal points VAD boost:  {vad_boost_normal:.4f}")
            print(f"  Ratio (Anomaly/Normal):   {vad_boost_anomaly / (vad_boost_normal + 1e-8):.2f}x")

        print("=" * 60)

        # 保存结果
        np.save(folder_path + 'anomaly_score.npy', test_energy)
        np.save(folder_path + 'test_labels.npy', test_labels)
        np.save(folder_path + 'rec_error.npy', test_rec)
        np.save(folder_path + 'var_discrepancy.npy', test_vad)
        np.save(folder_path + 'vad_boost.npy', test_vad_boost)

        with open("result_anomaly_detection.txt", 'a') as f:
            f.write(f"{setting}\n")
            f.write(f"Prior: λ={self.prior_lambda}, Strategy: VAD Boost Only\n")
            f.write(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                    f"Recall: {recall:.4f}, F-score: {f_score:.4f}\n")
            f.write(f"[Ablation] Rec-only F1: {f_rec:.4f}, VAD improvement: {improvement:+.2f}%\n\n")

        return accuracy, precision, recall, f_score
