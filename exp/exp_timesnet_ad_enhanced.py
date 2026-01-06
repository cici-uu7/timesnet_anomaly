"""
增强版 TimesNet_AD 实验类
基于原版 Anomaly Transformer 的训练和测试逻辑改进：
1. Minimax 两阶段训练（与原版一致）
2. Softmax 加权异常分数（与原版一致）
3. 多层 Loss 聚合
"""

from exp.exp_timesnet_ad import Exp_TimesNet_AD
from models import TimesNet_AD_Enhanced
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from utils.tools import adjustment


class Exp_TimesNet_AD_Enhanced(Exp_TimesNet_AD):
    """
    增强版实验类
    关键改进：
    1. Minimax 两阶段训练：loss1.backward(retain_graph=True) + loss2.backward()
    2. Softmax 加权异常分数：metric = softmax(-series_loss - prior_loss)
    """
    def __init__(self, args):
        super().__init__(args)
        self.model_dict['TimesNet_AD_Enhanced'] = TimesNet_AD_Enhanced

    def _build_model(self):
        """构建增强版模型"""
        model = TimesNet_AD_Enhanced.Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _normalize_prior(self, prior_attn):
        """
        归一化 Prior 注意力（与原版 Anomaly Transformer 一致）
        prior_attn: [B, H, L, L]
        返回: [B, H, L, L] 归一化后的 Prior
        """
        # 沿最后一维求和并归一化
        prior_sum = torch.sum(prior_attn, dim=-1, keepdim=True)  # [B, H, L, 1]
        prior_normalized = prior_attn / (prior_sum + 1e-8)
        return prior_normalized

    def _compute_minimax_loss(self, all_series_attn, all_prior_attn):
        """
        计算 Minimax 损失（与原版 Anomaly Transformer 一致）

        返回:
            series_loss: 用于 loss1 = rec_loss - k * series_loss
            prior_loss: 用于 loss2 = rec_loss + k * prior_loss
        """
        num_layers = len(all_series_attn)
        total_series_loss = 0.0
        total_prior_loss = 0.0

        for layer_idx in range(num_layers):
            series_attn = all_series_attn[layer_idx]  # [B, H, L, L]
            prior_attn = all_prior_attn[layer_idx]    # [B, H, L, L]

            # 归一化 Prior（与原版一致）
            prior_normalized = self._normalize_prior(prior_attn)

            series_loss = 0.0
            prior_loss = 0.0
            num_heads = series_attn.shape[1]

            for h in range(num_heads):
                s_attn = series_attn[:, h, :, :]      # [B, L, L]
                p_attn = prior_normalized[:, h, :, :]  # [B, L, L]

                # Series Loss: KL(Series || Prior.detach()) + KL(Prior.detach() || Series)
                # 注意：Prior 在 series_loss 计算时 detach
                series_loss += (
                    torch.mean(self.my_kl_loss(s_attn, p_attn.detach())) +
                    torch.mean(self.my_kl_loss(p_attn.detach(), s_attn))
                )

                # Prior Loss: KL(Prior || Series.detach()) + KL(Series.detach() || Prior)
                # 注意：Series 在 prior_loss 计算时 detach
                prior_loss += (
                    torch.mean(self.my_kl_loss(p_attn, s_attn.detach())) +
                    torch.mean(self.my_kl_loss(s_attn.detach(), p_attn))
                )

            series_loss = series_loss / num_heads
            prior_loss = prior_loss / num_heads

            total_series_loss += series_loss
            total_prior_loss += prior_loss

        # 平均多层 Loss
        total_series_loss = total_series_loss / num_layers
        total_prior_loss = total_prior_loss / num_layers

        return total_series_loss, total_prior_loss

    def train(self, setting):
        """
        Minimax 两阶段训练（与原版 Anomaly Transformer 一致）

        关键：
        loss1 = rec_loss - k * series_loss  → 最大化 Series-Prior 差异
        loss2 = rec_loss + k * prior_loss   → 最小化 Prior-Series 差异

        loss1.backward(retain_graph=True)
        loss2.backward()
        optimizer.step()
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        import time
        from utils.tools import EarlyStopping

        time_now = time.time()
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        optimizer = self._select_optimizer()
        criterion = nn.MSELoss()

        k_val = getattr(self.args, 'k', 3.0)

        print(f"Training Enhanced TimesNet_AD (Minimax Strategy)")
        print(f"  - Layers: {self.args.e_layers}")
        print(f"  - Minimax k={k_val}")
        print(f"  - Two-phase backward: loss1.backward(retain_graph=True) + loss2.backward()")
        print("")

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_series_loss = []
            epoch_prior_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, _) in enumerate(train_loader):
                iter_count += 1
                optimizer.zero_grad()

                batch_x = batch_x.float().to(self.device)

                # 前向传播
                output, all_series_attn, all_prior_attn = self.model(batch_x, None, None, None)

                # 1. 重构损失
                rec_loss = criterion(output, batch_x)

                # 2. Minimax 损失
                series_loss, prior_loss = self._compute_minimax_loss(
                    all_series_attn, all_prior_attn
                )

                # 3. 两阶段反向传播（与原版 Anomaly Transformer 一致）
                # Phase 1: 最大化 Series-Prior 差异
                loss1 = rec_loss - k_val * series_loss
                loss1.backward(retain_graph=True)

                # Phase 2: 最小化 Prior-Series 差异
                loss2 = rec_loss + k_val * prior_loss
                loss2.backward()

                optimizer.step()

                train_loss.append(rec_loss.item())
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
            avg_series_loss = np.average(epoch_series_loss)
            avg_prior_loss = np.average(epoch_prior_loss)

            # 验证
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            # 获取 Sigma 监控值
            sigma_means = [block.prior_sigma.mean().item() for block in self.model.anomaly_blocks]

            print(f"Epoch: {epoch + 1}, Steps: {len(train_loader)} | Train Loss: {train_loss_avg:.7f} Vali Loss: {vali_loss:.7f}")
            print(f"  ↳ S-Loss: {avg_series_loss:.4f}, P-Loss: {avg_prior_loss:.4f}, Sigma: {[f'{s:.3f}' for s in sigma_means]}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            from utils.tools import adjust_learning_rate
            adjust_learning_rate(optimizer, epoch + 1, self.args)

        # 训练结束
        sigma_final = [block.prior_sigma.mean().item() for block in self.model.anomaly_blocks]
        print(f"\n✓ Training complete. Final Sigma: {[f'{s:.3f}' for s in sigma_final]}")

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def _compute_anomaly_score_batch(self, batch_x, outputs, all_series_attn, all_prior_attn, temperature=50):
        """
        计算单个 batch 的异常分数（与原版 Anomaly Transformer 一致）

        关键：
        metric = softmax((-series_loss - prior_loss), dim=-1)
        anomaly_score = metric * rec_loss

        返回:
            anomaly_score: [B, L] - 每个时间点的异常分数
        """
        B, L, C = batch_x.shape
        num_layers = len(all_series_attn)

        # 1. 计算重构误差 [B, L]
        rec_loss = torch.mean((outputs - batch_x) ** 2, dim=-1)  # [B, L]

        # 2. 计算多层关联差异
        total_series_loss = 0.0
        total_prior_loss = 0.0

        for layer_idx in range(num_layers):
            series_attn = all_series_attn[layer_idx]  # [B, H, L, L]
            prior_attn = all_prior_attn[layer_idx]    # [B, H, L, L]

            # 归一化 Prior
            prior_normalized = self._normalize_prior(prior_attn)

            num_heads = series_attn.shape[1]

            for h in range(num_heads):
                s_attn = series_attn[:, h, :, :]      # [B, L, L]
                p_attn = prior_normalized[:, h, :, :]  # [B, L, L]

                if h == 0 and layer_idx == 0:
                    # 第一层第一头：初始化
                    series_loss = self.my_kl_loss(s_attn, p_attn.detach()) * temperature
                    prior_loss = self.my_kl_loss(p_attn, s_attn.detach()) * temperature
                else:
                    # 累加
                    series_loss = series_loss + self.my_kl_loss(s_attn, p_attn.detach()) * temperature
                    prior_loss = prior_loss + self.my_kl_loss(p_attn, s_attn.detach()) * temperature

        # series_loss, prior_loss: [B, L]

        # 3. Softmax 加权（与原版一致）
        # metric 越大表示该时间点 Series-Prior 差异越大（越可能是异常）
        metric = torch.softmax((-series_loss - prior_loss), dim=-1)  # [B, L]

        # 4. 加权重构误差
        anomaly_score = metric * rec_loss  # [B, L]

        return anomaly_score, series_loss, prior_loss

    def test(self, setting, test=0):
        """
        测试函数（与原版 Anomaly Transformer 一致的异常分数计算）
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
        temperature = 50
        anomaly_ratio = getattr(self.args, 'anomaly_ratio', 1.0)

        # ========== 训练集统计 ==========
        print("Computing anomaly scores on training set...")
        train_energy_list = []

        with torch.no_grad():
            for i, (batch_x, _) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                outputs, all_series_attn, all_prior_attn = self.model(batch_x, None, None, None)

                anomaly_score, _, _ = self._compute_anomaly_score_batch(
                    batch_x, outputs, all_series_attn, all_prior_attn, temperature
                )
                train_energy_list.append(anomaly_score.detach().cpu().numpy())

        train_energy = np.concatenate(train_energy_list, axis=0).reshape(-1)
        print(f"Train scores computed: {train_energy.shape[0]} points")

        # ========== 阈值集统计 ==========
        print("Computing threshold on test set (for threshold selection)...")
        thre_energy_list = []

        with torch.no_grad():
            for i, (batch_x, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                outputs, all_series_attn, all_prior_attn = self.model(batch_x, None, None, None)

                anomaly_score, _, _ = self._compute_anomaly_score_batch(
                    batch_x, outputs, all_series_attn, all_prior_attn, temperature
                )
                thre_energy_list.append(anomaly_score.detach().cpu().numpy())

        thre_energy = np.concatenate(thre_energy_list, axis=0).reshape(-1)

        # 阈值选择（与原版一致）
        combined_energy = np.concatenate([train_energy, thre_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - anomaly_ratio)
        print(f"Threshold: {threshold:.6f} (anomaly_ratio={anomaly_ratio}%)")

        # ========== 测试集评估 ==========
        print("Evaluating on test set...")
        test_energy_list = []
        test_labels_list = []

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                outputs, all_series_attn, all_prior_attn = self.model(batch_x, None, None, None)

                anomaly_score, series_loss, prior_loss = self._compute_anomaly_score_batch(
                    batch_x, outputs, all_series_attn, all_prior_attn, temperature
                )

                test_energy_list.append(anomaly_score.detach().cpu().numpy())
                test_labels_list.append(batch_y.numpy())

        test_energy = np.concatenate(test_energy_list, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels_list, axis=0).reshape(-1)

        print(f"Test data: {test_energy.shape[0]} points, Labels: {test_labels.shape[0]} points")

        # 统计信息
        print(f"\nEnergy Stats:")
        print(f"  Train: mean={train_energy.mean():.6f}, std={train_energy.std():.6f}")
        print(f"  Test:  mean={test_energy.mean():.6f}, std={test_energy.std():.6f}")

        # 预测
        pred = (test_energy > threshold).astype(int)
        gt = test_labels.astype(int)

        print(f"pred shape: {pred.shape}, gt shape: {gt.shape}")

        # Point Adjustment（与原版一致）
        gt, pred = adjustment(gt, pred)

        # 评估
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(gt, pred, average='binary')

        print(f"\n{'='*60}")
        print(f"Final Results:")
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  F1 Score : {f_score:.4f}")
        print(f"{'='*60}\n")

        # 保存结果
        np.save(folder_path + 'anomaly_score.npy', test_energy)
        np.save(folder_path + 'test_labels.npy', test_labels)

        return accuracy, precision, recall, f_score
