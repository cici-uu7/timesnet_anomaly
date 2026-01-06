"""
增强版 TimesNet_AD 实验类
支持：
1. 多层 Loss 聚合
2. 动态 Prior 训练
3. 多层级异常分数融合
"""

from exp.exp_timesnet_ad import Exp_TimesNet_AD
from models import TimesNet_AD_Enhanced
import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from utils.tools import adjustment


class Exp_TimesNet_AD_Enhanced(Exp_TimesNet_AD):
    """
    增强版实验类，基于原版扩展多层训练和测试逻辑
    """
    def __init__(self, args):
        # 先调用父类初始化
        super().__init__(args)
        # 注册增强版模型
        self.model_dict['TimesNet_AD_Enhanced'] = TimesNet_AD_Enhanced

    def _build_model(self):
        """构建增强版模型"""
        model = TimesNet_AD_Enhanced.Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _compute_multi_layer_loss(self, all_series_attn, all_prior_attn, k_val, margin_val):
        """
        计算多层Loss
        输入:
            all_series_attn: List of [B, H, L, L]
            all_prior_attn: List of [B, H, L, L] or [H, L, L]
            k_val: Minimax权重
            margin_val: 差异阈值
        返回:
            total_series_loss, total_prior_margin_loss
        """
        num_layers = len(all_series_attn)
        total_series_loss = 0.0
        total_prior_margin_loss = 0.0

        # 调试：记录每层的原始prior_loss
        debug_prior_losses = []

        for layer_idx in range(num_layers):
            series_attn = all_series_attn[layer_idx]
            prior_attn = all_prior_attn[layer_idx]

            series_loss = 0.0
            prior_loss = 0.0

            # 处理Prior维度
            if prior_attn.dim() == 3:  # [H, L, L]
                # 静态Prior，扩展batch维度
                prior_attn = prior_attn.unsqueeze(0).expand(series_attn.shape[0], -1, -1, -1)

            # 遍历每个Head
            num_heads = prior_attn.shape[1]
            for h in range(num_heads):
                s_attn = series_attn[:, h, :, :]
                p_attn = prior_attn[:, h, :, :]

                # Series Loss: KL(Series || Prior)
                series_loss += torch.mean(self.my_kl_loss(s_attn, p_attn))

                # Prior Loss: KL(Prior || Series)
                prior_loss += torch.mean(self.my_kl_loss(p_attn, s_attn))

            series_loss = series_loss / num_heads
            prior_loss = prior_loss / num_heads

            # 调试：记录原始prior_loss
            debug_prior_losses.append(prior_loss.item())

            # 新策略: 保持Prior独立性的连续损失函数
            # -prior_loss: 鼓励Prior与Series差异
            # 二次惩罚: 防止差异过大，稳定训练
            target_discrepancy = margin_val
            prior_margin_loss = -prior_loss + 0.5 * (prior_loss - target_discrepancy) ** 2

            total_series_loss += series_loss
            total_prior_margin_loss += prior_margin_loss

        # 平均多层Loss
        total_series_loss = total_series_loss / num_layers
        total_prior_margin_loss = total_prior_margin_loss / num_layers

        # 🔍 调试输出：第一次调用时打印原始prior_loss
        if not hasattr(self, '_debug_printed'):
            import numpy as np
            print(f"\n🔍 First batch debug:")
            print(f"   Per-layer prior_loss: {[f'{x:.4f}' for x in debug_prior_losses]}")
            print(f"   Average: {np.mean(debug_prior_losses):.4f}")
            print(f"   Margin: {margin_val}")
            print(f"   Margin - avg: {margin_val - np.mean(debug_prior_losses):.4f}")
            print(f"   → If negative, old clamp would return 0!\n")
            self._debug_printed = True

        return total_series_loss, total_prior_margin_loss

    def train(self, setting):
        """
        增强版训练函数
        支持多层Loss聚合
        """
        # 获取数据加载器
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        import time
        from utils.tools import EarlyStopping

        time_now = time.time()
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # 初始化早停机制
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        optimizer = self._select_optimizer()
        criterion = nn.MSELoss()

        # 获取参数
        k_val = getattr(self.args, 'k', 3.0)
        margin_val = getattr(self.args, 'margin', 0.5)
        dynamic_prior = getattr(self.args, 'dynamic_prior', True)

        print(f"Training Enhanced TimesNet_AD")
        print(f"  - Layers: {self.args.e_layers}, Fusion: {getattr(self.args, 'fusion_method', 'weighted')}, Dynamic Prior: {dynamic_prior}")
        print(f"  - Minimax: k={k_val}, margin={margin_val}")

        # 只在第一个epoch前打印初始Sigma
        sigma_means = [block.prior_sigma.data.mean().item() for block in self.model.anomaly_blocks]
        print(f"  - Initial Sigma: {sigma_means}")
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

                # 前向传播：返回重构结果 + 多层注意力
                output, all_series_attn, all_prior_attn = self.model(batch_x, None, None, None)

                # 1. 重构损失
                rec_loss = criterion(output, batch_x)

                # 2. 多层关联差异损失
                series_loss, prior_margin_loss = self._compute_multi_layer_loss(
                    all_series_attn, all_prior_attn, k_val, margin_val
                )

                # 3. 总损失
                loss = rec_loss + k_val * series_loss + k_val * prior_margin_loss

                # 反向传播
                loss.backward()
                optimizer.step()

                train_loss.append(rec_loss.item())
                epoch_series_loss.append(series_loss.item())
                epoch_prior_loss.append(prior_margin_loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {rec_loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * len(train_loader) - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            avg_series_loss = np.average(epoch_series_loss)
            avg_prior_loss = np.average(epoch_prior_loss)

            # 验证集验证
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            # 获取当前Sigma均值（用于监控）
            sigma_means = [block.prior_sigma.data.mean().item() for block in self.model.anomaly_blocks]

            print(f"Epoch: {epoch + 1}, Steps: {len(train_loader)} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}")
            print(f"  ↳ S-Loss: {avg_series_loss:.4f}, P-Loss: {avg_prior_loss:.4f}, Sigma: {[f'{s:.2f}' for s in sigma_means]}")

            # 早停判断
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            from utils.tools import adjust_learning_rate
            adjust_learning_rate(optimizer, epoch + 1, self.args)

        # 训练结束，简洁输出最终Sigma
        sigma_final = [block.prior_sigma.data.mean().item() for block in self.model.anomaly_blocks]
        print(f"\n✓ Training complete. Final Sigma: {[f'{s:.2f}' for s in sigma_final]}")

        # 加载最优模型
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def _compute_multi_layer_anomaly_score(self, all_rec_errors, all_assoc_discs, alpha, beta):
        """
        计算多层级异常分数
        输入:
            all_rec_errors: List of [B, L] - 每层的重构误差
            all_assoc_discs: List of [B, L] - 每层的关联差异
            alpha, beta: 权重
        返回:
            combined_score: [B, L]
        """
        num_layers = len(all_rec_errors)

        # 方法1: 简单平均
        avg_rec_error = sum(all_rec_errors) / num_layers
        avg_assoc_disc = sum(all_assoc_discs) / num_layers

        # 归一化
        rec_mean, rec_std = avg_rec_error.mean(), avg_rec_error.std()
        assoc_mean, assoc_std = avg_assoc_disc.mean(), avg_assoc_disc.std()

        rec_error_norm = (avg_rec_error - rec_mean) / (rec_std + 1e-8)
        assoc_disc_norm = (avg_assoc_disc - assoc_mean) / (assoc_std + 1e-8)

        # 加权组合
        combined_score = alpha * rec_error_norm + beta * assoc_disc_norm

        return combined_score

    def test(self, setting, test=0):
        """
        增强版测试函数
        支持多层级异常分数融合
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

        # 获取参数
        alpha = getattr(self.args, 'alpha', 0.5)
        beta = getattr(self.args, 'beta', 0.5)
        anomaly_ratio = getattr(self.args, 'anomaly_ratio', 1.0)

        print("Computing anomaly scores on training set (normal data)...")
        train_rec_errors_list = [[] for _ in range(self.args.e_layers)]
        train_assoc_discs_list = [[] for _ in range(self.args.e_layers)]

        with torch.no_grad():
            for i, (batch_x, _) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)

                # 前向传播
                outputs, all_series_attn, all_prior_attn = self.model(batch_x, None, None, None)

                # 计算每层的异常分数
                for layer_idx in range(self.args.e_layers):
                    series_attn = all_series_attn[layer_idx]
                    prior_attn = all_prior_attn[layer_idx]

                    # 重构误差
                    rec_error = torch.mean((outputs - batch_x) ** 2, dim=-1)

                    # 关联差异
                    if prior_attn.dim() == 3:  # [H, L, L]
                        prior_attn = prior_attn.unsqueeze(0).expand(series_attn.shape[0], -1, -1, -1)

                    kl_div = 0.0
                    for h in range(series_attn.shape[1]):
                        s_attn = series_attn[:, h, :, :]
                        p_attn = prior_attn[:, h, :, :]
                        kl_div += (self.my_kl_loss(s_attn, p_attn) + self.my_kl_loss(p_attn, s_attn)) / 2.0

                    assoc_disc = kl_div / series_attn.shape[1]

                    train_rec_errors_list[layer_idx].append(rec_error.detach().cpu().numpy())
                    train_assoc_discs_list[layer_idx].append(assoc_disc.detach().cpu().numpy())

        # 合并训练集分数
        train_rec_errors_layers = [np.concatenate(train_rec_errors_list[i], axis=0).reshape(-1)
                                    for i in range(self.args.e_layers)]
        train_assoc_discs_layers = [np.concatenate(train_assoc_discs_list[i], axis=0).reshape(-1)
                                     for i in range(self.args.e_layers)]

        print(f"Train scores computed: {train_rec_errors_layers[0].shape[0]} points")

        # 测试集
        print("Computing anomaly scores on test set...")
        test_rec_errors_list = [[] for _ in range(self.args.e_layers)]
        test_assoc_discs_list = [[] for _ in range(self.args.e_layers)]
        test_labels = []

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)

                # 前向传播
                outputs, all_series_attn, all_prior_attn = self.model(batch_x, None, None, None)

                # 计算每层的异常分数
                for layer_idx in range(self.args.e_layers):
                    series_attn = all_series_attn[layer_idx]
                    prior_attn = all_prior_attn[layer_idx]

                    # 重构误差
                    rec_error = torch.mean((outputs - batch_x) ** 2, dim=-1)

                    # 关联差异
                    if prior_attn.dim() == 3:
                        prior_attn = prior_attn.unsqueeze(0).expand(series_attn.shape[0], -1, -1, -1)

                    kl_div = 0.0
                    for h in range(series_attn.shape[1]):
                        s_attn = series_attn[:, h, :, :]
                        p_attn = prior_attn[:, h, :, :]
                        kl_div += (self.my_kl_loss(s_attn, p_attn) + self.my_kl_loss(p_attn, s_attn)) / 2.0

                    assoc_disc = kl_div / series_attn.shape[1]

                    test_rec_errors_list[layer_idx].append(rec_error.detach().cpu().numpy())
                    test_assoc_discs_list[layer_idx].append(assoc_disc.detach().cpu().numpy())

                test_labels.append(batch_y.numpy())

        # 合并测试集分数
        test_rec_errors_layers = [np.concatenate(test_rec_errors_list[i], axis=0).reshape(-1)
                                   for i in range(self.args.e_layers)]
        test_assoc_discs_layers = [np.concatenate(test_assoc_discs_list[i], axis=0).reshape(-1)
                                    for i in range(self.args.e_layers)]
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)

        print(f"Test data shapes - rec: {test_rec_errors_layers[0].shape}, labels: {test_labels.shape}")

        # 简洁的每层统计（一行显示）
        print("\nLayer Stats (Train/Test Assoc):")
        layer_stats = []
        for i in range(self.args.e_layers):
            train_assoc = train_assoc_discs_layers[i].mean()
            test_assoc = test_assoc_discs_layers[i].mean()
            layer_stats.append(f"L{i+1}: {train_assoc:.3f}/{test_assoc:.3f}")
        print("  " + ", ".join(layer_stats))

        # 多层级融合异常分数
        # 简单平均
        train_rec_avg = np.mean(train_rec_errors_layers, axis=0)
        train_assoc_avg = np.mean(train_assoc_discs_layers, axis=0)
        test_rec_avg = np.mean(test_rec_errors_layers, axis=0)
        test_assoc_avg = np.mean(test_assoc_discs_layers, axis=0)

        # 联合归一化
        all_rec = np.concatenate([train_rec_avg, test_rec_avg], axis=0)
        all_assoc = np.concatenate([train_assoc_avg, test_assoc_avg], axis=0)

        rec_mean, rec_std = all_rec.mean(), all_rec.std()
        assoc_mean, assoc_std = all_assoc.mean(), all_assoc.std()

        print(f"Combined Stats - Rec: {rec_mean:.4f}±{rec_std:.4f}, Assoc: {assoc_mean:.4f}±{assoc_std:.4f}")

        # ⚠️ 关键诊断：如果Assoc < 0.1，立即警告
        if assoc_mean < 0.1:
            print(f"⚠️  WARNING: Association Discrepancy too low ({assoc_mean:.4f})! Prior-Series collapsed!")
        elif assoc_mean > 1.5:
            print(f"⚠️  WARNING: Association Discrepancy too high ({assoc_mean:.4f})! Training unstable!")

        # 归一化
        train_rec_norm = (train_rec_avg - rec_mean) / (rec_std + 1e-8)
        train_assoc_norm = (train_assoc_avg - assoc_mean) / (assoc_std + 1e-8)
        test_rec_norm = (test_rec_avg - rec_mean) / (rec_std + 1e-8)
        test_assoc_norm = (test_assoc_avg - assoc_mean) / (assoc_std + 1e-8)

        # 加权组合
        train_energy = alpha * train_rec_norm + beta * train_assoc_norm
        test_energy = alpha * test_rec_norm + beta * test_assoc_norm

        # 阈值选择
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - anomaly_ratio)

        print(f"Threshold: {threshold:.4f} (α={alpha}, β={beta}, anomaly_ratio={anomaly_ratio}%)")

        # 预测
        pred = (test_energy > threshold).astype(int)
        gt = test_labels.astype(int)

        # Point Adjustment
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

        return
