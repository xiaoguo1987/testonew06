# -*- coding: utf-8 -*-
"""
PPO（近端策略优化）训练配置
集中管理 PPO 超参数，便于调参与复现。

RTX 4090D GPU流水线优化版 v2：
- 网络架构升级 512-512-256-128（约800K参数），充分利用24GB显存
- 增大batch_size（512）提升GPU利用率
- 降低update_epochs（4）避免策略退化
- 增加warmup_steps（3000）稳定初期训练
- 新增KL散度早停配置
- 增大total_training_steps（300000）
- GPU批量推理桥 + 异步流水线训练

SEED RL GPU-Centric 架构（--arch seed-rl）：
- 大模型 hidden_dims=[2048,2048,1024,512]（约14M参数）
- 集中式 GPU 推理 + PPO 训练
- CPU worker 仅跑游戏逻辑
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class PPOConfig:
    """PPO 模型与训练超参数
    
    RTX 4090D GPU流水线优化 v2：
    - 更大网络（512-512-256-128）充分压榨4090D算力
    - 更大批次（512）以充分利用4090D算力
    - 更少更新轮次（4）减少策略退化风险
    - KL散度早停机制
    - 学习率warmup和cosine衰减
    - 自适应熵系数
    - 价值损失裁剪
    - 梯度累积
    - GPU批量推理 + 异步训练流水线
    """

    # 网络结构
    input_dim: int = 134  # 与 StateRepresentation.total_state_dim 一致（6-Max）
    output_dim: int = 5    # 动作数：fold, call, raise_small, raise_mid, raise_allin

    # 学习率
    lr: float = 3e-4
    lr_min: float = 1e-6
    lr_decay_rate: float = 0.9999
    critic_lr_multiplier: float = 3.0  # Critic 学习率倍率（相对 Actor），Critic 需要更快收敛以提供准确的价值估计
    
    # 学习率调度优化
    use_lr_warmup: bool = True
    warmup_steps: int = 3000
    use_cosine_decay: bool = True
    total_training_steps: int = 300000

    # PPO 核心参数
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    lambda_: float = 0.95
    
    # 熵正则优化
    entropy_coef: float = 0.02
    entropy_adaptive: bool = True
    entropy_coef_target: float = 0.01
    
    # 价值损失裁剪
    value_clip: float = 0.2

    # KL早停阈值（从0.5提高到2.0：避免低概率动作的ratio爆炸触发过早KL早停，
    # 导致raise2x/4x等动作永远无法获得梯度更新）
    kl_target: float = 2.0
    kl_early_stop: bool = True

    # 每批轨迹更新轮次（减少同一 buffer 的重复使用，避免策略偏移过大）
    update_epochs: int = 4
    batch_size: int = 256
    
    # 梯度累积
    gradient_accumulation_steps: int = 1

    # 梯度裁剪（防止爆炸）
    max_grad_norm: float = 0.5

    # SEED RL 大模型配置
    hidden_dims: List[int] = field(default_factory=lambda: [512, 512, 256, 128])
    residual_layer_idx: int = 1  # 残差连接层索引
    inference_batch_size: int = 256  # GPU 推理批量大小
    inference_timeout_ms: float = 1.0  # 推理收集超时（毫秒）
    ppo_buffer_size: int = 16384  # SEED RL 触发 PPO 训练的最小样本数（16K 步让大模型看清全局）

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
            "use_lr_warmup": self.use_lr_warmup,
            "warmup_steps": self.warmup_steps,
            "use_cosine_decay": self.use_cosine_decay,
            "entropy_adaptive": self.entropy_adaptive,
            "value_clip": self.value_clip,
            "kl_target": self.kl_target,
            "kl_early_stop": self.kl_early_stop,
            "critic_lr_multiplier": self.critic_lr_multiplier,
            "hidden_dims": self.hidden_dims,
            "residual_layer_idx": self.residual_layer_idx,
            "inference_batch_size": self.inference_batch_size,
            "ppo_buffer_size": self.ppo_buffer_size,
        }


# 默认配置实例（可由 train.py 或 SelfPlayTrainer 覆盖）
DEFAULT_PPO_CONFIG = PPOConfig()
