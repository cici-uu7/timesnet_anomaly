"""
Exp_TimesNet_AD_Bayesian: MC Dropout贝叶斯不确定性估计实验类

核心创新：
1. 训练：标准的重构损失（与原版TimesNet相同）
2. 测试：MC Dropout多次采样，计算预测方差作为不确定性
3. 异常分数：重构误差 + 不确定性加权

优势：
- 无需修改训练过程
- 测试时自动估计模型不确定性
- 适用于所有数据集（PSM、龙门铣、SMD等）
"""

from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from models import TimesNet_AD_Bayesian
import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from utils.tools import adjustment


class Exp_TimesNet_AD_Bayesian(Exp_Anomaly_Detection):
    """
    贝叶斯TimesNet异常检测实验类

    关键方法：
    - train(): 标准训练（继承自父类）
    - test(): MC Dropout测试
    - test_with_mc_dropout(): 核心MC Dropout推理逻辑
    """

    def __init__(self, args):
        super().__init__(args)
        self.model_dict['TimesNet_AD_Bayesian'] = TimesNet_AD_Bayesian

        # MC Dropout参数
        self.mc_samples = getattr(args, 'mc_samples', 5)  # 默认采样5次
        self.uncertainty_weight = getattr(args, 'uncertainty_weight', 0.5)  # 不确定性权重
        self.pretrained_model = getattr(args, 'pretrained_model', None)  # 预训练模型路径

    def _build_model(self):
        """构建贝叶斯TimesNet模型"""
        model = TimesNet_AD_Bayesian.Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _mc_dropout_inference(self, batch_x, mc_samples=5):
        """
        MC Dropout推理：多次前向传播，计算均值和方差

        Args:
            batch_x: [B, L, C] 输入数据
            mc_samples: MC采样次数

        Returns:
            mean_pred: [B, L, C] 预测均值
            uncertainty: [B, L] 预测方差（不确定性）
        """
        # 关键：保持模型在训练模式（Dropout开启）
        self.model.train()

        predictions = []

        with torch.no_grad():
            for _ in range(mc_samples):
                # 每次前向传播，Dropout的mask不同
                pred = self.model(batch_x, None, None, None)
                predictions.append(pred)

        # 堆叠所有预测 [mc_samples, B, L, C]
        predictions = torch.stack(predictions, dim=0)

        # 计算均值和方差
        mean_pred = predictions.mean(dim=0)  # [B, L, C]
        variance = predictions.var(dim=0)    # [B, L, C]

        # 对变量维度取平均，得到每个时间点的总体不确定性
        uncertainty = variance.mean(dim=-1)  # [B, L]

        return mean_pred, uncertainty

    def _compute_anomaly_score(self, outputs, inputs, uncertainty, strategy='weighted'):
        """
        计算异常分数：重构误差 + 不确定性

        Args:
            outputs: [B, L, C] 预测输出
            inputs: [B, L, C] 真实输入
            uncertainty: [B, L] 不确定性
            strategy: 融合策略 ('weighted', 'additive', 'multiplicative')

        Returns:
            anomaly_score: [B, L] 异常分数
        """
        # 1. 计算重构误差
        rec_error = ((outputs - inputs) ** 2).mean(dim=-1)  # [B, L]

        # 2. 归一化不确定性（避免数值不稳定）
        uncertainty_norm = (uncertainty - uncertainty.mean()) / (uncertainty.std() + 1e-8)
        uncertainty_norm = torch.clamp(uncertainty_norm, -3, 3)  # 截断异常值

        # 3. 融合策略
        if strategy == 'additive':
            # 策略1：加法融合
            score = rec_error + self.uncertainty_weight * uncertainty
        elif strategy == 'multiplicative':
            # 策略2：乘法融合
            score = rec_error * (1 + self.uncertainty_weight * torch.relu(uncertainty_norm))
        else:  # weighted (默认)
            # 策略3：加权融合（推荐）
            score = rec_error + self.uncertainty_weight * uncertainty

        return score

    def test(self, setting, test=0):
        """
        MC Dropout测试函数

        核心流程：
        1. 在训练集上计算阈值
        2. 在测试集上用MC Dropout推理
        3. 融合重构误差和不确定性
        4. 评估性能
        """
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')

        if test:
            print('Loading model...')
            # 优先使用预训练模型路径，否则使用setting路径
            if self.pretrained_model and os.path.exists(self.pretrained_model):
                model_path = self.pretrained_model
                print(f'Using pretrained model: {model_path}')
            else:
                model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
                print(f'Using checkpoint: {model_path}')

            self.model.load_state_dict(torch.load(model_path))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        anomaly_ratio = getattr(self.args, 'anomaly_ratio', 1.0)
        strategy = getattr(self.args, 'score_strategy', 'weighted')

        print("=" * 60)
        print(f"MC Dropout Bayesian Uncertainty Estimation")
        print(f"  - MC Samples: {self.mc_samples}")
        print(f"  - Uncertainty Weight: {self.uncertainty_weight}")
        print(f"  - Score Strategy: {strategy}")
        print(f"  - Anomaly Ratio: {anomaly_ratio}%")
        print("=" * 60)

        # ========== 第1步：训练集统计 ==========
        print("\n[Step 1/4] Computing scores on training set...")
        train_scores = []
        train_uncertainties = []
        train_rec_errors = []  # 直接保存重构误差

        with torch.no_grad():
            for i, (batch_x, _) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)

                # MC Dropout推理
                mean_pred, uncertainty = self._mc_dropout_inference(batch_x, self.mc_samples)

                # 计算异常分数
                score = self._compute_anomaly_score(mean_pred, batch_x, uncertainty, strategy)

                # 纯重构误差（用于消融实验）
                rec_error = ((mean_pred - batch_x) ** 2).mean(dim=-1)

                train_scores.append(score.detach().cpu().numpy())
                train_uncertainties.append(uncertainty.detach().cpu().numpy())
                train_rec_errors.append(rec_error.detach().cpu().numpy())

        train_scores = np.concatenate(train_scores, axis=0).reshape(-1)
        train_uncertainties = np.concatenate(train_uncertainties, axis=0).reshape(-1)
        train_rec_errors = np.concatenate(train_rec_errors, axis=0).reshape(-1)

        print(f"Train scores: {train_scores.shape[0]} points")
        print(f"  Score: mean={train_scores.mean():.6f}, std={train_scores.std():.6f}")
        print(f"  Uncertainty: mean={train_uncertainties.mean():.6f}, std={train_uncertainties.std():.6f}")
        print(f"  Rec Error: mean={train_rec_errors.mean():.6f}, std={train_rec_errors.std():.6f}")

        # ========== 第2步：测试集推理 ==========
        print("\n[Step 2/4] Computing scores on test set...")
        test_scores = []
        test_uncertainties = []
        test_labels = []
        test_rec_errors = []  # 保存纯重构误差用于对比

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.float().to(self.device)

                # MC Dropout推理
                mean_pred, uncertainty = self._mc_dropout_inference(batch_x, self.mc_samples)

                # 计算异常分数
                score = self._compute_anomaly_score(mean_pred, batch_x, uncertainty, strategy)

                # 纯重构误差（用于消融实验）
                rec_error = ((mean_pred - batch_x) ** 2).mean(dim=-1)

                test_scores.append(score.detach().cpu().numpy())
                test_uncertainties.append(uncertainty.detach().cpu().numpy())
                test_rec_errors.append(rec_error.detach().cpu().numpy())
                test_labels.append(batch_y.numpy())

        test_scores = np.concatenate(test_scores, axis=0).reshape(-1)
        test_uncertainties = np.concatenate(test_uncertainties, axis=0).reshape(-1)
        test_rec_errors = np.concatenate(test_rec_errors, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)

        print(f"Test scores: {test_scores.shape[0]} points")
        print(f"  Score: mean={test_scores.mean():.6f}, std={test_scores.std():.6f}")
        print(f"  Uncertainty: mean={test_uncertainties.mean():.6f}, std={test_uncertainties.std():.6f}")
        print(f"  Rec Error: mean={test_rec_errors.mean():.6f}, std={test_rec_errors.std():.6f}")

        # ========== 第3步：计算阈值 ==========
        print("\n[Step 3/4] Computing threshold...")
        combined_scores = np.concatenate([train_scores, test_scores], axis=0)
        threshold = np.percentile(combined_scores, 100 - anomaly_ratio)
        print(f"Threshold: {threshold:.6f} (anomaly_ratio={anomaly_ratio}%)")

        # ========== 第4步：评估性能 ==========
        print("\n[Step 4/4] Evaluating performance...")

        # 预测
        pred = (test_scores > threshold).astype(int)
        gt = test_labels.astype(int)

        # Point Adjustment
        gt, pred = adjustment(gt, pred)

        # 计算指标
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            gt, pred, average='binary'
        )

        print("\n" + "=" * 60)
        print("Results (MC Dropout + Uncertainty):")
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  F1 Score : {f_score:.4f}")
        print("=" * 60)

        # ========== 消融实验：纯重构误差 ==========
        print("\n[Ablation] Reconstruction Error Only (without Uncertainty):")

        # 使用纯重构误差计算阈值（直接使用保存的重构误差，而非反推）
        combined_rec_errors = np.concatenate([train_rec_errors, test_rec_errors], axis=0)
        threshold_rec = np.percentile(combined_rec_errors, 100 - anomaly_ratio)

        pred_rec = (test_rec_errors > threshold_rec).astype(int)
        gt_rec = test_labels.astype(int)
        gt_rec, pred_rec = adjustment(gt_rec, pred_rec)

        _, _, f_score_rec, _ = precision_recall_fscore_support(
            gt_rec, pred_rec, average='binary'
        )

        print(f"  F1 (Rec only): {f_score_rec:.4f}")

        improvement = ((f_score - f_score_rec) / f_score_rec * 100) if f_score_rec > 0 else 0
        print(f"  Uncertainty improvement: {improvement:+.2f}%")

        # ========== 不确定性分析 ==========
        print("\n" + "=" * 60)
        print("[Analysis] Uncertainty Statistics:")

        # 分析正常点和异常点的不确定性
        normal_uncertainty = test_uncertainties[test_labels == 0]
        anomaly_uncertainty = test_uncertainties[test_labels == 1]

        print(f"  Normal points uncertainty:")
        print(f"    mean={normal_uncertainty.mean():.6f}, std={normal_uncertainty.std():.6f}")
        print(f"  Anomaly points uncertainty:")
        print(f"    mean={anomaly_uncertainty.mean():.6f}, std={anomaly_uncertainty.std():.6f}")

        # 计算不确定性的区分度（AUC）
        from sklearn.metrics import roc_auc_score
        try:
            uncertainty_auc = roc_auc_score(test_labels, test_uncertainties)
            print(f"  Uncertainty AUC: {uncertainty_auc:.4f}")
            if uncertainty_auc > 0.6:
                print(f"  ✓ Uncertainty has discriminative power")
            else:
                print(f"  ✗ Uncertainty has low discriminative power")
        except:
            print(f"  Unable to compute AUC")

        print("=" * 60)

        # ========== 保存结果 ==========
        print("\nSaving results...")
        np.save(folder_path + 'anomaly_score.npy', test_scores)
        np.save(folder_path + 'uncertainty.npy', test_uncertainties)
        np.save(folder_path + 'rec_error.npy', test_rec_errors)
        np.save(folder_path + 'test_labels.npy', test_labels)

        print(f"Results saved to {folder_path}")

        return accuracy, precision, recall, f_score
