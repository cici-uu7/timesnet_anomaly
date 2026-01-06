"""
TimesNet_AD_V2 实验类
最后层融合架构: TimesNet 重构 + AnomalyAttention 关联差异

核心设计:
1. 异常分数 = α * 重构误差 + β * 关联差异
2. 两种信号独立计算，不互相干扰
3. 可调节 α, β 权重来平衡两种检测机制
"""

from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from models import TimesNet_AD_V2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from utils.tools import adjustment, EarlyStopping, adjust_learning_rate


class Exp_TimesNet_AD_V2(Exp_Basic):
    """
    TimesNet_AD_V2 实验类

    关键特性:
    1. 纯 TimesNet 重构 (前 N 层全部是 TimesBlock)
    2. 最后添加 Anomaly Attention (仅用于关联差异)
    3. 异常分数 = α * rec_error + β * assoc_discrepancy
    """
    def __init__(self, args):
        super().__init__(args)
        self.model_dict = {'TimesNet_AD_V2': TimesNet_AD_V2}

    def _build_model(self):
        """构建模型"""
        model = TimesNet_AD_V2.Model(self.args).float()
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

    def my_kl_loss(self, p, q):
        """KL 散度 (与 Anomaly Transformer 一致)"""
        res = p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))
        return torch.sum(res, dim=-1)

    def _normalize_prior(self, prior_attn):
        """归一化 Prior 注意力"""
        prior_sum = torch.sum(prior_attn, dim=-1, keepdim=True)
        prior_normalized = prior_attn / (prior_sum + 1e-8)
        return prior_normalized

    def _compute_association_discrepancy(self, series_attn, prior_attn):
        """
        计算关联差异

        Args:
            series_attn: [B, H, L, L]
            prior_attn: [B, H, L, L]
        Returns:
            discrepancy: [B, L] 每个时间步的关联差异
        """
        prior_normalized = self._normalize_prior(prior_attn)
        num_heads = series_attn.shape[1]

        # 计算每个 head 的 KL 散度
        discrepancy = 0.0
        for h in range(num_heads):
            s_attn = series_attn[:, h, :, :]  # [B, L, L]
            p_attn = prior_normalized[:, h, :, :]  # [B, L, L]

            # 双向 KL 散度
            kl_sp = self.my_kl_loss(s_attn, p_attn)  # [B, L]
            kl_ps = self.my_kl_loss(p_attn, s_attn)  # [B, L]

            discrepancy = discrepancy + kl_sp + kl_ps

        discrepancy = discrepancy / num_heads
        return discrepancy

    def _compute_minimax_loss(self, series_attn, prior_attn):
        """
        计算 Minimax 损失 (用于训练)
        """
        prior_normalized = self._normalize_prior(prior_attn)
        num_heads = series_attn.shape[1]

        series_loss = 0.0
        prior_loss = 0.0

        for h in range(num_heads):
            s_attn = series_attn[:, h, :, :]
            p_attn = prior_normalized[:, h, :, :]

            # Series Loss (Series 试图拉近 Prior)
            series_loss += (
                torch.mean(self.my_kl_loss(s_attn, p_attn.detach())) +
                torch.mean(self.my_kl_loss(p_attn.detach(), s_attn))
            )

            # Prior Loss (Prior 试图拉近 Series)
            prior_loss += (
                torch.mean(self.my_kl_loss(p_attn, s_attn.detach())) +
                torch.mean(self.my_kl_loss(s_attn.detach(), p_attn))
            )

        series_loss = series_loss / num_heads
        prior_loss = prior_loss / num_heads

        return series_loss, prior_loss

    def vali(self, vali_data, vali_loader, criterion):
        """验证"""
        self.model.eval()
        total_loss = []

        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                outputs, series_attn, prior_attn = self.model(batch_x, None, None, None)
                loss = criterion(outputs, batch_x)
                total_loss.append(loss.item())

        avg_loss = np.average(total_loss)
        self.model.train()
        return avg_loss

    def train(self, setting):
        """
        训练函数

        训练策略:
        1. 主要优化重构损失 (保持 TimesNet 的重构能力)
        2. 辅助 Minimax 训练 (让关联差异有区分能力)
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

        k_val = getattr(self.args, 'k', 1.0)  # 默认较小的 k，保护重构能力

        print(f"Training TimesNet_AD_V2 (Last-Layer Fusion)")
        print(f"  - Architecture: {self.args.e_layers} TimesBlock layers + 1 AnomalyAttention")
        print(f"  - Minimax k={k_val} (small k to protect reconstruction)")
        print(f"  - Training: rec_loss + k*(series_loss + prior_loss)")
        print("")

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_rec_loss = []
            epoch_series_loss = []
            epoch_prior_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, _) in enumerate(train_loader):
                iter_count += 1
                optimizer.zero_grad()

                batch_x = batch_x.float().to(self.device)

                # 前向传播
                outputs, series_attn, prior_attn = self.model(batch_x, None, None, None)

                # 1. 重构损失 (主要目标)
                rec_loss = criterion(outputs, batch_x)

                # 2. Minimax 损失 (辅助目标)
                series_loss, prior_loss = self._compute_minimax_loss(series_attn, prior_attn)

                # 总损失: 重构为主，Minimax 为辅
                # 不使用两阶段训练，避免干扰重构学习
                total_loss = rec_loss + k_val * 0.1 * (series_loss + prior_loss)

                total_loss.backward()
                optimizer.step()

                train_loss.append(total_loss.item())
                epoch_rec_loss.append(rec_loss.item())
                epoch_series_loss.append(series_loss.item())
                epoch_prior_loss.append(prior_loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | rec_loss: {rec_loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * len(train_loader) - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")
            train_loss_avg = np.average(train_loss)
            avg_rec_loss = np.average(epoch_rec_loss)
            avg_series_loss = np.average(epoch_series_loss)
            avg_prior_loss = np.average(epoch_prior_loss)

            # 验证
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {len(train_loader)} | Train Loss: {train_loss_avg:.7f} Vali Loss: {vali_loss:.7f}")
            print(f"  ↳ Rec: {avg_rec_loss:.6f}, S-Loss: {avg_series_loss:.4f}, P-Loss: {avg_prior_loss:.4f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(optimizer, epoch + 1, self.args)

        print(f"\n✓ Training complete.")

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        """
        测试函数

        异常分数 = α * rec_error + β * assoc_discrepancy
        """
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()

        # 异常分数权重
        alpha = getattr(self.args, 'alpha', 0.5)  # 重构误差权重
        beta = getattr(self.args, 'beta', 0.5)    # 关联差异权重
        anomaly_ratio = getattr(self.args, 'anomaly_ratio', 1.0)

        print(f"Anomaly Score = {alpha} * rec_error + {beta} * assoc_discrepancy")

        # ========== 训练集统计 ==========
        print("Computing anomaly scores on training set...")
        train_rec_list = []
        train_assoc_list = []

        with torch.no_grad():
            for i, (batch_x, _) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                outputs, series_attn, prior_attn = self.model(batch_x, None, None, None)

                # 重构误差
                rec_error = torch.mean((outputs - batch_x) ** 2, dim=-1)  # [B, L]

                # 关联差异
                assoc_disc = self._compute_association_discrepancy(series_attn, prior_attn)  # [B, L]

                train_rec_list.append(rec_error.cpu().numpy())
                train_assoc_list.append(assoc_disc.cpu().numpy())

        train_rec = np.concatenate(train_rec_list, axis=0).reshape(-1)
        train_assoc = np.concatenate(train_assoc_list, axis=0).reshape(-1)

        # 归一化
        rec_mean, rec_std = train_rec.mean(), train_rec.std() + 1e-8
        assoc_mean, assoc_std = train_assoc.mean(), train_assoc.std() + 1e-8

        print(f"Train Rec: mean={rec_mean:.6f}, std={rec_std:.6f}")
        print(f"Train Assoc: mean={assoc_mean:.6f}, std={assoc_std:.6f}")

        # 训练集异常分数
        train_rec_norm = (train_rec - rec_mean) / rec_std
        train_assoc_norm = (train_assoc - assoc_mean) / assoc_std
        train_energy = alpha * train_rec_norm + beta * train_assoc_norm

        print(f"Train scores computed: {train_energy.shape[0]} points")

        # ========== 测试集评估 ==========
        print("Evaluating on test set...")
        test_rec_list = []
        test_assoc_list = []
        test_labels_list = []

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                outputs, series_attn, prior_attn = self.model(batch_x, None, None, None)

                # 重构误差
                rec_error = torch.mean((outputs - batch_x) ** 2, dim=-1)

                # 关联差异
                assoc_disc = self._compute_association_discrepancy(series_attn, prior_attn)

                test_rec_list.append(rec_error.cpu().numpy())
                test_assoc_list.append(assoc_disc.cpu().numpy())
                test_labels_list.append(batch_y.numpy())

        test_rec = np.concatenate(test_rec_list, axis=0).reshape(-1)
        test_assoc = np.concatenate(test_assoc_list, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels_list, axis=0).reshape(-1)

        # 归一化 (使用训练集统计量)
        test_rec_norm = (test_rec - rec_mean) / rec_std
        test_assoc_norm = (test_assoc - assoc_mean) / assoc_std
        test_energy = alpha * test_rec_norm + beta * test_assoc_norm

        print(f"Test data: {test_energy.shape[0]} points")

        # ========== 阈值选择 ==========
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - anomaly_ratio)
        print(f"Threshold: {threshold:.6f} (anomaly_ratio={anomaly_ratio}%)")

        # ========== 统计信息 ==========
        print(f"\nScore Stats:")
        print(f"  Train: mean={train_energy.mean():.6f}, std={train_energy.std():.6f}")
        print(f"  Test:  mean={test_energy.mean():.6f}, std={test_energy.std():.6f}")
        print(f"  Test Rec (norm): mean={test_rec_norm.mean():.4f}, std={test_rec_norm.std():.4f}")
        print(f"  Test Assoc (norm): mean={test_assoc_norm.mean():.4f}, std={test_assoc_norm.std():.4f}")

        # ========== 预测 ==========
        pred = (test_energy > threshold).astype(int)
        gt = test_labels.astype(int)

        print(f"pred shape: {pred.shape}, gt shape: {gt.shape}")

        # Point Adjustment
        gt, pred = adjustment(gt, pred)

        # ========== 评估 ==========
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(gt, pred, average='binary')

        print(f"\n{'='*60}")
        print(f"Final Results (alpha={alpha}, beta={beta}):")
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  F1 Score : {f_score:.4f}")
        print(f"{'='*60}\n")

        # 保存结果
        np.save(folder_path + 'anomaly_score.npy', test_energy)
        np.save(folder_path + 'test_labels.npy', test_labels)
        np.save(folder_path + 'rec_error.npy', test_rec)
        np.save(folder_path + 'assoc_disc.npy', test_assoc)

        return accuracy, precision, recall, f_score
