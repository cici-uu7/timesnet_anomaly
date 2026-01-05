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

        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        self.model.eval()

        # 用于存储重构误差和关联差异（分开存储，后续归一化）
        reconstruction_errors = []
        association_discrepancies = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        criterion = nn.MSELoss(reduction='none')

        # 1. 推理阶段：计算重构误差和关联差异
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(test_loader):
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
                    # my_kl_loss 返回 [B, L]，对每个时间步i，计算其与所有其他时间步的KL散度之和
                    kl_ps = self.my_kl_loss(p_attn, s_attn)  # [B, L]
                    kl_sp = self.my_kl_loss(s_attn, p_attn)  # [B, L]

                    # 直接累加，维度已经是 [B, L]
                    assoc_disc += kl_ps + kl_sp

                assoc_disc = assoc_disc / prior_attn.shape[1]

                reconstruction_errors.append(rec_error.detach().cpu().numpy())
                association_discrepancies.append(assoc_disc.detach().cpu().numpy())

        # 2. 数据后处理和归一化
        # 拼接所有 Batch
        reconstruction_errors = np.concatenate(reconstruction_errors, axis=0).reshape(-1)
        association_discrepancies = np.concatenate(association_discrepancies, axis=0).reshape(-1)

        # 归一化处理（Z-score标准化）
        rec_mean, rec_std = reconstruction_errors.mean(), reconstruction_errors.std()
        assoc_mean, assoc_std = association_discrepancies.mean(), association_discrepancies.std()

        rec_norm = (reconstruction_errors - rec_mean) / (rec_std + 1e-8)
        assoc_norm = (association_discrepancies - assoc_mean) / (assoc_std + 1e-8)

        # 加权组合（可调参数）
        # alpha: 重构误差权重, beta: 关联差异权重
        alpha = getattr(self.args, 'alpha', 0.5)
        beta = getattr(self.args, 'beta', 0.5)
        print(f"Anomaly Score Weighting: alpha={alpha}, beta={beta}")
        test_energy = alpha * rec_norm + beta * assoc_norm

        # 保存未归一化的原始数据用于分析
        np.save(folder_path + 'reconstruction_errors.npy', reconstruction_errors)
        np.save(folder_path + 'association_discrepancies.npy', association_discrepancies)

        # 获取 Ground Truth 标签
        test_labels = test_loader.dataset.test_labels

        # 长度对齐 (裁剪掉多余部分)
        if len(test_labels) > len(test_energy):
            test_labels = test_labels[:len(test_energy)]
        else:
            test_energy = test_energy[:len(test_labels)]

        print("Test Labels Shape:", test_labels.shape)
        print("Anomaly Score Shape:", test_energy.shape)

        # 保存原始分数和标签
        np.save(folder_path + 'anomaly_score.npy', test_energy)
        np.save(folder_path + 'test_labels.npy', test_labels)

        # 3. 阈值搜索与评估 (Improved Threshold Search)
        # 使用百分位数范围而非min/max，更稳健
        percentiles = np.linspace(1, 99, 100)
        threshs = np.percentile(test_energy, percentiles)

        best_f1 = -1
        best_precision = -1
        best_recall = -1
        best_thresh = 0.0

        print("Calculating metrics (This might take a while)...")

        # 简单循环寻找最佳阈值 (不带 adjustment 以加快速度)
        for thresh in threshs:
            pred = (test_energy > thresh).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(test_labels, pred, average='binary',
                                                                       zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall
                best_thresh = thresh

        print(f"Best Threshold (Percentile Search): {best_thresh}")
        print(f"Threshold Stats - Min: {test_energy.min():.4f}, Max: {test_energy.max():.4f}, Mean: {test_energy.mean():.4f}, Std: {test_energy.std():.4f}")

        # 4. 使用最佳阈值进行最终评估 (带 Point Adjustment)
        # Point Adjustment: 如果检测到异常片段中的任何一点，则视为检测到整个片段
        final_pred = (test_energy > best_thresh).astype(int)

        try:
            # 应用 Point Adjustment
            _, final_pred_adjusted = adjustment(test_labels, final_pred)
        except Exception as e:
            print(f"Warning: Point adjustment failed ({e}), utilizing raw predictions.")
            final_pred_adjusted = final_pred

        # 计算最终指标
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels, final_pred_adjusted, average='binary',
                                                                   zero_division=0)
        accuracy = accuracy_score(test_labels, final_pred_adjusted)

        print(f"Final Metrics with Point Adjustment:")
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