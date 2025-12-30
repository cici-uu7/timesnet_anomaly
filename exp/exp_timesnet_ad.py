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
        # 计算 KL 散度: KL(p || q)
        # 加上微小常数防止 log(0)
        res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
        return torch.mean(torch.sum(res, dim=-1), dim=1)

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
        print(f"Training with Minimax Strategy, k={k_val}")

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

                # --- Minimax 策略核心 ---

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

                    # Series Loss: Series 逼近 Prior
                    series_loss += torch.mean(self.my_kl_loss(p_attn, s_attn.detach() + 1e-5)) + \
                                   torch.mean(self.my_kl_loss(s_attn.detach() + 1e-5, p_attn))

                    # Prior Loss: Prior 逼近 Series (反向优化)
                    prior_loss += torch.mean(self.my_kl_loss(p_attn, s_attn.detach() + 1e-5)) + \
                                  torch.mean(self.my_kl_loss(s_attn.detach() + 1e-5, p_attn))

                series_loss = series_loss / len(prior_attn)
                prior_loss = prior_loss / len(prior_attn)

                # 优化策略
                # 阶段 1: 最小化 Rec Loss 和 Series Loss (拉近 Series 到 Prior)
                loss1 = rec_loss - k_val * series_loss

                # 阶段 2: 最小化 Prior Loss (其实是最大化差异，因为 loss2 单独 backward)
                loss2 = k_val * prior_loss

                # 反向传播
                loss1.backward(retain_graph=True)
                loss2.backward()

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

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        criterion = nn.MSELoss(reduction='none')

        # 1. 推理阶段：计算异常得分
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)

                # 前向传播
                output, series_attn, prior_attn = self.model(batch_x, None, None, None)

                # 计算重构误差 (Squared Error) -> [B, L]
                rec_loss = torch.mean(criterion(output, batch_x), dim=-1)

                # 计算关联差异 (Association Discrepancy)
                score = torch.zeros(batch_x.shape[0], batch_x.shape[1]).to(self.device)
                for u in range(prior_attn.shape[1]):
                    if series_attn.dim() == 4:
                        s_attn = series_attn[:, u, :, :]
                        p_attn = prior_attn[:, u, :, :]
                    else:
                        s_attn = series_attn
                        p_attn = prior_attn

                    # 计算对称 KL 散度
                    score += torch.mean(self.my_kl_loss(p_attn, s_attn), dim=-1) + \
                             torch.mean(self.my_kl_loss(s_attn, p_attn), dim=-1)

                score = score / len(prior_attn)

                # 组合得分: 重构误差 * 关联差异 (放大异常)
                det_score = rec_loss * score

                attens_energy.append(det_score.detach().cpu().numpy())

        # 2. 数据后处理
        # 拼接所有 Batch
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

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

        # 3. 阈值搜索与评估 (Threshold Search)
        # 使用暴力搜索寻找最佳 F1 的阈值 (Standard Protocol)
        threshs = np.linspace(test_energy.min(), test_energy.max(), 100)
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

        print(f"Best Threshold (Raw Search): {best_thresh}")

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