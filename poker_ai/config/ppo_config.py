# -*- coding: utf-8 -*-
"""
PPO（近端策略优化）训练配置
集中管理 PPO 超参数，便于调参与复现。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PPOConfig:
    """PPO 模型与训练超参数"""

    # 网络结构
    input_dim: int = 134  # 与 StateRepresentation.total_state_dim 一致（6-Max）
    output_dim: int = 5    # 动作数：fold, call, raise_small, raise_mid, raise_allin

    # 学习率
    lr: float = 3e-4
    lr_min: float = 1e-6
    lr_decay_rate: float = 0.9999

    # PPO 核心参数
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    lambda_: float = 0.95
    entropy_coef: float = 0.02

    # 每批轨迹更新轮次与批次
    update_epochs: int = 10
    batch_size: int = 64

    # 梯度裁剪（防止爆炸）
    max_grad_norm: float = 0.5

    def to_dict(self):
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "lr": self.lr,
            "clip_epsilon": self.clip_epsilon,
            "gamma": self.gamma,
            "lambda_": self.lambda_,
            "entropy_coef": self.entropy_coef,
            "update_epochs": self.update_epochs,
            "batch_size": self.batch_size,
        }


# 默认配置实例（可由 train.py 或 SelfPlayTrainer 覆盖）
DEFAULT_PPO_CONFIG = PPOConfig()
