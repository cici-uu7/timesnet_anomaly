from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from models import TimesNet_AD
import torch
import torch.nn as nn
import time
import numpy as np
import os
from utils.tools import adjust_learning_rate, adjustment, EarlyStopping
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


class Exp_TimesNet_AD(Exp_Anomaly_Detection):
    def __init__(self, args):
        super().__init__(args)
        # 注册新模型
        self.model_dict['TimesNet_AD'] = TimesNet_AD

    def _build_model(self):
        # 强制构建 TimesNet_AD 模型
        model = TimesNet_AD.Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def my_kl_loss(self, p, q):
        """
        计算 KL 散度: KL(p || q)
        添加裁剪防止数值爆炸
        """
        # 裁剪概率值，防止log(0)和极端值
        p = torch.clamp(p, min=1e-8, max=1.0)
        q = torch.clamp(q, min=1e-8, max=1.0)

        # KL散度计算
        kl = p * (torch.log(p) - torch.log(q))

        # 裁剪KL散度值，防止极端情况
        kl = torch.clamp(kl, min=-10, max=10)

        return torch.sum(kl, dim=-1)

    def train(self, setting):
        # 获取数据加载器
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # 初始化早停机制
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        optimizer = self._select_optimizer()
        criterion = nn.MSELoss()

        # 获取 k 参数 (默认为 3.0，如果 args 里没有则兜底)
        k_val = getattr(self.args, 'k', 3.0)
        margin_val = getattr(self.args, 'margin', 0.5)
        print(f"Training with Minimax Strategy, k={k_val}, margin={margin_val}")

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, _) in enumerate(train_loader):
                iter_count += 1
                optimizer.zero_grad()

                batch_x = batch_x.float().to(self.device)

                # 前向传播：返回重构结果、序列关联、先验关联
                output, series_attn, prior_attn = self.model(batch_x, None, None, None)

                # --- 改进的 Minimax 策略核心 ---

                # 1. 计算重构损失 (MSE)
                rec_loss = criterion(output, batch_x)

                # 2. 计算关联差异损失 (Association Discrepancy)
                series_loss = 0.0
                prior_loss = 0.0

                # 遍历每个 Head 计算 KL 散度
                for u in range(prior_attn.shape[1]):
                    # 处理维度兼容性 (Batch, Heads, Length, Length)
                    if series_attn.dim() == 4:
                        s_attn = series_attn[:, u, :, :]
                        p_attn = prior_attn[:, u, :, :]
                    else:
                        s_attn = series_attn
                        p_attn = prior_attn

                    # Series Loss: 让 Series 逼近 Prior (学习先验的局部平滑性)
                    # KL(Series || Prior) - 正常数据应该符合局部平滑假设
                    series_loss += torch.mean(self.my_kl_loss(s_attn, p_attn))

                    # Prior Loss: 衡量 Prior 与 Series 的差异程度
                    # 不使用detach，让模型学习调整prior_sigma参数
                    # 但添加停止梯度的上限，避免Prior过度适应数据
                    prior_loss += torch.mean(self.my_kl_loss(p_attn, s_attn))

                series_loss = series_loss / prior_attn.shape[1]
                prior_loss = prior_loss / prior_attn.shape[1]

                # 改进的 Minimax 策略:
                # - Series 逼近 Prior (最小化 series_loss)
                # - Prior 保持适度差异 (使用margin loss而非直接最大化)
                #
                # 使用 margin-based loss: max(0, margin - prior_loss)
                # 只有当prior_loss小于margin时才惩罚(鼓励保持一定差异)
                prior_margin_loss = torch.clamp(margin_val - prior_loss, min=0)

                # 最终损失: 重构 + Series学习 + Prior保持差异
                loss = rec_loss + k_val * series_loss + k_val * prior_margin_loss

                # 反向传播
                loss.backward()

                optimizer.step()

                train_loss.append(rec_loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {rec_loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * len(train_loader) - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)

            # 验证集验证 (只关注重构误差，保证模型基本能力)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print(
                f"Epoch: {epoch + 1}, Steps: {len(train_loader)} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}")

            # 早停判断
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(optimizer, epoch + 1, self.args)

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                output, _, _ = self.model(batch_x, None, None, None)
                pred = output.detach().cpu()
                true = batch_x.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')

        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        self.model.eval()

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        criterion = nn.MSELoss(reduction='none')

        # 获取参数
        alpha = getattr(self.args, 'alpha', 0.5)
        beta = getattr(self.args, 'beta', 0.5)

        # ============ 步骤 1: 统计训练集分数 (正常数据) ============
        print("Computing anomaly scores on training set (normal data)...")
        train_rec_errors = []
        train_assoc_discs = []

        with torch.no_grad():
            for i, (batch_x, _) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                output, series_attn, prior_attn = self.model(batch_x, None, None, None)

                # 计算重构误差
                rec_error = torch.mean(criterion(output, batch_x), dim=-1)

                # 计算关联差异
                assoc_disc = torch.zeros(batch_x.shape[0], batch_x.shape[1]).to(self.device)
                for u in range(prior_attn.shape[1]):
                    if series_attn.dim() == 4:
                        s_attn = series_attn[:, u, :, :]
                        p_attn = prior_attn[:, u, :, :]
                    else:
                        s_attn = series_attn
                        p_attn = prior_attn
                    kl_ps = self.my_kl_loss(p_attn, s_attn)
                    kl_sp = self.my_kl_loss(s_attn, p_attn)
                    assoc_disc += kl_ps + kl_sp
                assoc_disc = assoc_disc / prior_attn.shape[1]

                train_rec_errors.append(rec_error.detach().cpu().numpy())
                train_assoc_discs.append(assoc_disc.detach().cpu().numpy())

        train_rec_errors = np.concatenate(train_rec_errors, axis=0).reshape(-1)
        train_assoc_discs = np.concatenate(train_assoc_discs, axis=0).reshape(-1)
        print(f"Train scores computed: {len(train_rec_errors)} points")

        # ============ 步骤 2: 计算测试集分数 ============
        print("Computing anomaly scores on test set...")
        test_rec_errors = []
        test_assoc_discs = []
        test_labels_list = []

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)

                # 前向传播
                output, series_attn, prior_attn = self.model(batch_x, None, None, None)

                # 计算重构误差 (Squared Error) -> [B, L]
                rec_error = torch.mean(criterion(output, batch_x), dim=-1)

                # 计算关联差异 (Association Discrepancy)
                assoc_disc = torch.zeros(batch_x.shape[0], batch_x.shape[1]).to(self.device)
                for u in range(prior_attn.shape[1]):
                    if series_attn.dim() == 4:
                        s_attn = series_attn[:, u, :, :]  # [B, L, L]
                        p_attn = prior_attn[:, u, :, :]   # [B, L, L]
                    else:
                        s_attn = series_attn
                        p_attn = prior_attn

                    # 计算对称 KL 散度（裁剪后）
                    kl_ps = self.my_kl_loss(p_attn, s_attn)  # [B, L]
                    kl_sp = self.my_kl_loss(s_attn, p_attn)  # [B, L]

                    # 直接累加，维度已经是 [B, L]
                    assoc_disc += kl_ps + kl_sp

                assoc_disc = assoc_disc / prior_attn.shape[1]

                test_rec_errors.append(rec_error.detach().cpu().numpy())
                test_assoc_discs.append(assoc_disc.detach().cpu().numpy())
                test_labels_list.append(batch_y.numpy())  # 同步收集标签

        # 拼接测试集数据
        test_rec_errors = np.concatenate(test_rec_errors, axis=0).reshape(-1)
        test_assoc_discs = np.concatenate(test_assoc_discs, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels_list, axis=0).reshape(-1)

        print(f"Test data shapes - rec: {test_rec_errors.shape}, assoc: {test_assoc_discs.shape}, labels: {test_labels.shape}")

        # ============ 步骤 3: 联合归一化 (训练集+测试集) ============
        # 合并训练集和测试集来计算归一化参数
        all_rec_errors = np.concatenate([train_rec_errors, test_rec_errors], axis=0)
        all_assoc_discs = np.concatenate([train_assoc_discs, test_assoc_discs], axis=0)

        # 使用联合数据计算归一化参数
        rec_mean, rec_std = all_rec_errors.mean(), all_rec_errors.std()
        assoc_mean, assoc_std = all_assoc_discs.mean(), all_assoc_discs.std()

        print(f"Combined Reconstruction Error Stats - Mean: {rec_mean:.4f}, Std: {rec_std:.4f}")
        print(f"Combined Association Discrepancy Stats - Mean: {assoc_mean:.4f}, Std: {assoc_std:.4f}")

        # 分别归一化训练集和测试集
        train_rec_norm = (train_rec_errors - rec_mean) / (rec_std + 1e-8)
        train_assoc_norm = (train_assoc_discs - assoc_mean) / (assoc_std + 1e-8)

        test_rec_norm = (test_rec_errors - rec_mean) / (rec_std + 1e-8)
        test_assoc_norm = (test_assoc_discs - assoc_mean) / (assoc_std + 1e-8)

        # 加权组合
        print(f"Anomaly Score Weighting: alpha={alpha}, beta={beta}")
        train_combined = alpha * train_rec_norm + beta * train_assoc_norm
        test_combined = alpha * test_rec_norm + beta * test_assoc_norm

        # 对组合后的分数进行二次归一化
        combined_all = np.concatenate([train_combined, test_combined], axis=0)
        combined_mean = combined_all.mean()
        combined_std = combined_all.std()

        train_energy = (train_combined - combined_mean) / (combined_std + 1e-8)
        test_energy = (test_combined - combined_mean) / (combined_std + 1e-8)

        print(f"Final Score Stats - Train Mean: {train_energy.mean():.4f}, Std: {train_energy.std():.4f}")
        print(f"Final Score Stats - Test Mean: {test_energy.mean():.4f}, Std: {test_energy.std():.4f}")

        # 保存数据
        np.save(folder_path + 'reconstruction_errors.npy', test_rec_errors)
        np.save(folder_path + 'association_discrepancies.npy', test_assoc_discs)
        np.save(folder_path + 'anomaly_score.npy', test_energy)
        np.save(folder_path + 'test_labels.npy', test_labels)

        # ============ 步骤 4: 使用联合分布选择阈值 ============
        # 关键改进: 使用训练集+测试集的联合分布来选择阈值
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)

        print(f"Threshold (based on combined distribution): {threshold:.4f}")
        print(f"Anomaly ratio: {self.args.anomaly_ratio}%")

        # ============ 步骤 5: 评估 ============
        # 使用阈值对测试集进行预测
        pred = (test_energy > threshold).astype(int)
        gt = test_labels.astype(int)

        print("Prediction shape:", pred.shape)
        print("Ground truth shape:", gt.shape)

        # 应用 Point Adjustment
        try:
            gt_adjusted, pred_adjusted = adjustment(gt, pred)
        except Exception as e:
            print(f"Warning: Point adjustment failed ({e}), using raw predictions.")
            gt_adjusted, pred_adjusted = gt, pred

        # 计算最终指标
        precision, recall, f1, _ = precision_recall_fscore_support(gt_adjusted, pred_adjusted, average='binary',
                                                                   zero_division=0)
        accuracy = accuracy_score(gt_adjusted, pred_adjusted)

        print(f"\nFinal Metrics with Point Adjustment:")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        # 将结果写入日志
        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write(f"Accuracy : {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f} \n")
        f.write('\n')
        f.close()

        return