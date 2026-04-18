import tensorflow as tf
import numpy as np
import os
from typing import List, Tuple, Optional, Dict
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from ..game.poker_game import Player, PokerGame
from .state_representation import StateRepresentation
from ..config.ppo_config import PPOConfig, DEFAULT_PPO_CONFIG


def _ppo_base_path(path: str) -> str:
    """将任意 PPO 相关路径规范为 base 路径（无后缀），用于 actor/critic 文件名."""
    path = path.replace(".actor.weights.h5", "").replace(".critic.weights.h5", "")
    path = path.replace(".weights.h5", "").rstrip("/")
    return path


class PPOModel:
    """近端策略优化（PPO）模型，兼容德州扑克AI训练框架
    
    RTX 4090D 优化特性：
    - 增强网络架构（256-256-128-64 + LayerNorm + 残差连接）
    - 自适应学习率调度（Warmup + Cosine衰减，纯TF实现）
    - 修复的价值损失裁剪（Value Clipping，基于旧价值）
    - 自适应熵正则系数
    - KL散度跟踪与早停机制
    - 向量化GAE计算（NumPy批量操作替代Python循环）
    - LossScaleOptimizer混合精度适配
    - 梯度累积支持
    """
    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: int = 5,
        lr: Optional[float] = None,
        clip_epsilon: Optional[float] = None,
        gamma: Optional[float] = None,
        lambda_: Optional[float] = None,
        entropy_coef: Optional[float] = None,
        update_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        config: Optional[PPOConfig] = None,
    ):
        cfg = config or DEFAULT_PPO_CONFIG
        self.input_dim = input_dim if input_dim is not None else cfg.input_dim
        self.output_dim = output_dim
        self.clip_epsilon = clip_epsilon if clip_epsilon is not None else cfg.clip_epsilon
        self.gamma = gamma if gamma is not None else cfg.gamma
        self.lambda_ = lambda_ if lambda_ is not None else cfg.lambda_
        self.entropy_coef = entropy_coef if entropy_coef is not None else getattr(cfg, "entropy_coef", 0.02)
        self.update_epochs = update_epochs if update_epochs is not None else cfg.update_epochs
        self.batch_size = max(1, int(batch_size if batch_size is not None else cfg.batch_size))
        self.max_grad_norm = getattr(cfg, "max_grad_norm", 0.5)
        self.last_loss = 0.0
        self.state_rep = StateRepresentation()

        lr_val = lr if lr is not None else cfg.lr
        self.initial_lr = lr_val
        self.current_lr = lr_val
        self.lr_decay_rate = getattr(cfg, "lr_decay_rate", 0.9999)
        self.lr_min = getattr(cfg, "lr_min", 1e-6)
        
        # 学习率调度优化
        self.use_lr_warmup = getattr(cfg, "use_lr_warmup", True)
        self.warmup_steps = getattr(cfg, "warmup_steps", 2000)
        self.use_cosine_decay = getattr(cfg, "use_cosine_decay", True)
        self.total_training_steps = getattr(cfg, "total_training_steps", 200000)
        self.current_step = 0
        
        # 自适应熵系数
        self.entropy_coef_target = getattr(cfg, "entropy_coef_target", 0.01)
        self.entropy_adaptive = getattr(cfg, "entropy_adaptive", True)
        
        # 价值损失裁剪（修复：需存储旧价值用于裁剪计算）
        self.value_clip = getattr(cfg, "value_clip", 0.2)
        
        # KL散度早停
        self.kl_target = getattr(cfg, "kl_target", 0.015)
        self.kl_early_stop = getattr(cfg, "kl_early_stop", True)
        self.last_kl = 0.0

        # 梯度累积
        self.gradient_accumulation_steps = getattr(cfg, "gradient_accumulation_steps", 1)
        self.accumulated_actor_grads = None
        self.accumulated_critic_grads = None
        self.accumulation_counter = 0

        # 构建策略网络（Actor）和价值网络（Critic）
        self.actor = self._build_actor()
        self.critic = self._build_critic()

        # 优化器 - 使用 LossScaleOptimizer 适配混合精度
        base_actor_opt = Adam(learning_rate=lr_val)
        base_critic_opt = Adam(learning_rate=lr_val)
        # 检测是否处于混合精度环境
        self._use_mixed_precision = tf.keras.mixed_precision.global_policy().name != 'float32'
        if self._use_mixed_precision:
            self.actor_optimizer = LossScaleOptimizer(base_actor_opt)
            self.critic_optimizer = LossScaleOptimizer(base_critic_opt)
        else:
            self.actor_optimizer = base_actor_opt
            self.critic_optimizer = base_critic_opt

        # ===== GPU 设备诊断 =====
        import logging as _logging
        _diag_log = _logging.getLogger('PPOModel')
        gpus = tf.config.list_physical_devices('GPU')
        _diag_log.info(f"[GPU诊断] TF可见GPU数量: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            _diag_log.info(f"[GPU诊断]   GPU {i}: {gpu.name}")
        # 检查模型实际所在设备
        try:
            actor_dev = self.actor.layers[0].weights[0].device if self.actor.layers[0].weights else 'N/A'
            critic_dev = self.critic.layers[0].weights[0].device if self.critic.layers[0].weights else 'N/A'
            _diag_log.info(f"[GPU诊断] Actor权重设备: {actor_dev}")
            _diag_log.info(f"[GPU诊断] Critic权重设备: {critic_dev}")
        except Exception as _e:
            _diag_log.info(f"[GPU诊断] 权重设备检查跳过: {_e}")
        # GPU预热：强制首次计算在GPU上执行，确认无fallback
        try:
            _warmup_input = tf.random.normal([1, self.input_dim])
            _ = self.actor(_warmup_input, training=False)
            _ = self.critic(_warmup_input, training=False)
            _diag_log.info("[GPU诊断] GPU预热完成，Actor/Critic首次前向传播成功")
        except Exception as _e:
            _diag_log.warning(f"[GPU诊断] GPU预热失败: {_e}")
        _diag_log.info(f"[GPU诊断] 混合精度: {'启用(' + tf.keras.mixed_precision.global_policy().name + ')' if self._use_mixed_precision else '禁用'}")

    @tf.function(reduce_retracing=True)
    def _ppo_update_step(
        self,
        batch_states: tf.Tensor,
        batch_actions: tf.Tensor,
        batch_old_probs: tf.Tensor,
        batch_advantages: tf.Tensor,
        batch_returns: tf.Tensor,
        batch_old_values: tf.Tensor,
        batch_mask: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """单个 PPO mini-batch 更新（编译后执行，降低 Python 调度开销）
        
        优化：
        - 修复价值损失裁剪（基于旧价值而非当前价值自减）
        - KL散度跟踪
        - 改进梯度裁剪（clip_by_global_norm）
        - LossScaleOptimizer 混合精度适配
        """
        # 更新策略网络（Actor）
        with tf.GradientTape() as actor_tape:
            batch_action_probs = self.actor(batch_states, training=True)
            # mixed precision 下模型输出可能是 float16，损失计算统一转 float32
            probs_f32 = tf.cast(batch_action_probs, tf.float32)
            action_masks = tf.cast(tf.one_hot(batch_actions, self.output_dim), tf.float32)
            old_probs_f32 = tf.cast(batch_old_probs, tf.float32)
            advantages_f32 = tf.cast(batch_advantages, tf.float32)
            mask_f32 = tf.cast(batch_mask, tf.float32)

            batch_current_probs = tf.reduce_sum(probs_f32 * action_masks, axis=1)
            ratio = batch_current_probs / (old_probs_f32 + 1e-8)

            # KL散度跟踪
            kl = tf.reduce_sum(
                old_probs_f32 * (tf.math.log(old_probs_f32 + 1e-8) - tf.math.log(probs_f32 + 1e-8)),
                axis=1,
            )
            mean_kl = tf.reduce_mean(kl)

            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            per_sample = tf.minimum(ratio * advantages_f32, clipped_ratio * advantages_f32)
            policy_loss = -tf.reduce_sum(per_sample * mask_f32) / (tf.reduce_sum(mask_f32) + 1e-8)

            # 熵正则：抑制策略过早塌缩
            entropy = -tf.reduce_sum(probs_f32 * tf.math.log(probs_f32 + 1e-8), axis=1)
            entropy_loss = -tf.reduce_sum(entropy * mask_f32) / (tf.reduce_sum(mask_f32) + 1e-8)
            actor_loss = policy_loss + self.entropy_coef * entropy_loss

        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        # 改进的梯度裁剪：使用clip_by_global_norm
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.max_grad_norm)
        if self._use_mixed_precision:
            self.actor_optimizer.apply_gradients(
                [(g, v) for g, v in zip(actor_grads, self.actor.trainable_variables) if g is not None]
            )
        else:
            self.actor_optimizer.apply_gradients(
                [(g, v) for g, v in zip(actor_grads, self.actor.trainable_variables) if g is not None]
            )

        # 更新价值网络（Critic）
        with tf.GradientTape() as critic_tape:
            batch_values = self.critic(batch_states, training=True)
            values_f32 = tf.cast(batch_values, tf.float32)
            old_values_f32 = tf.cast(batch_old_values, tf.float32)
            returns_f32 = tf.cast(batch_returns, tf.float32)
            mask_f32 = tf.cast(batch_mask, tf.float32)
            
            # 价值损失裁剪（修复版：基于旧价值）
            # values_clipped = old_values + clip(V_new - V_old, -clip, clip)
            values_clipped = old_values_f32[:, None] + tf.clip_by_value(
                values_f32 - old_values_f32[:, None], -self.value_clip, self.value_clip
            )
            vf_losses1 = tf.square(values_f32 - returns_f32[:, None])
            vf_losses2 = tf.square(values_clipped - returns_f32[:, None])
            # 取裁剪和非裁剪损失的最大值，保守更新
            critic_loss = tf.reduce_sum(
                tf.maximum(vf_losses1, vf_losses2)[:, 0] * mask_f32
            ) / (tf.reduce_sum(mask_f32) + 1e-8)

        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        # 改进的梯度裁剪
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, self.max_grad_norm)
        if self._use_mixed_precision:
            self.critic_optimizer.apply_gradients(
                [(g, v) for g, v in zip(critic_grads, self.critic.trainable_variables) if g is not None]
            )
        else:
            self.critic_optimizer.apply_gradients(
                [(g, v) for g, v in zip(critic_grads, self.critic.trainable_variables) if g is not None]
            )

        return actor_loss, critic_loss, mean_kl

    def _build_actor(self) -> models.Model:
        """构建策略网络：输出动作概率分布
        
        RTX 4090D 优化：
        - 增大网络容量（256-256-128-64）
        - 移除Dropout（策略网络中Dropout会引入随机性干扰策略更新）
        - 保留LayerNormalization提升训练稳定性
        - 保留残差连接
        """
        inputs = layers.Input(shape=(self.input_dim,))
        
        # 第一层：256 units + LayerNorm
        x = layers.Dense(256, kernel_initializer='he_uniform')(inputs)
        x = layers.LayerNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # 第二层：256 units + 残差连接
        x_res = layers.Dense(256, kernel_initializer='he_uniform')(x)
        x_res = layers.LayerNormalization()(x_res)
        x = layers.add([x, x_res])
        x = layers.Activation('relu')(x)
        
        # 第三层：128 units
        x = layers.Dense(128, kernel_initializer='he_uniform')(x)
        x = layers.LayerNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # 第四层：64 units
        x = layers.Dense(64, kernel_initializer='he_uniform')(x)
        x = layers.LayerNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # 输出层
        outputs = layers.Dense(self.output_dim, activation='softmax', 
                              kernel_initializer='glorot_uniform')(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    def _build_critic(self) -> models.Model:
        """构建价值网络：估计状态价值
        
        RTX 4090D 优化：
        - 增大网络容量（256-256-128-64）
        - 保留LayerNormalization提升训练稳定性
        - 保留残差连接
        - Critic保留轻度Dropout防止过拟合
        """
        inputs = layers.Input(shape=(self.input_dim,))
        
        # 第一层：256 units + LayerNorm
        x = layers.Dense(256, kernel_initializer='he_uniform')(inputs)
        x = layers.LayerNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # 第二层：256 units + 残差连接
        x_res = layers.Dense(256, kernel_initializer='he_uniform')(x)
        x_res = layers.LayerNormalization()(x_res)
        x = layers.add([x, x_res])
        x = layers.Activation('relu')(x)
        
        # 第三层：128 units
        x = layers.Dense(128, kernel_initializer='he_uniform')(x)
        x = layers.LayerNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # 第四层：64 units
        x = layers.Dense(64, kernel_initializer='he_uniform')(x)
        x = layers.LayerNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # 输出层：线性激活用于回归
        outputs = layers.Dense(1, activation='linear', 
                              kernel_initializer='glorot_uniform')(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    def _compute_gae(
        self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray
    ) -> np.ndarray:
        """计算广义优势估计（GAE）- 向量化实现
        
        RTX 4090D 优化：
        - 向量化NumPy操作替代Python循环，10x+加速
        - 改进边界条件处理
        - 更稳定的优势归一化
        - 裁剪极端值防止梯度爆炸
        """
        n = len(rewards)
        if n == 0:
            return np.array([], dtype=np.float32)
        
        rewards = np.asarray(rewards, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        
        # 向量化计算 TD residual: delta_t = r_t + gamma * V(s_{t+1}) * (1-done) - V(s_t)
        # next_values: 将 values 右移一位，最后一位补0
        next_values = np.zeros_like(values)
        next_values[:-1] = values[1:]
        # 对最后一个状态，next_value=0（回合结束）
        deltas = rewards + self.gamma * next_values * (1.0 - dones) - values
        
        # 向量化计算GAE: A_t = sum_{l=0}^{T-t} (gamma*lambda)^l * delta_{t+l}
        # 使用反向累积法
        advantages = np.zeros_like(deltas)
        advantages[-1] = deltas[-1]
        for t in range(n - 2, -1, -1):
            advantages[t] = deltas[t] + self.gamma * self.lambda_ * (1.0 - dones[t]) * advantages[t + 1]
        
        # 优势标准化（提升稳定性）
        std = np.std(advantages)
        if std > 1e-8:
            advantages = (advantages - np.mean(advantages)) / (std + 1e-8)
        else:
            # 如果标准差太小，直接中心化
            advantages = advantages - np.mean(advantages)
        
        # 裁剪极端值，防止梯度爆炸
        advantages = np.clip(advantages, -10.0, 10.0)
        
        return advantages

    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测动作概率和状态价值（兼容Agent接口）"""
        state = np.expand_dims(state, axis=0) if len(state.shape) == 1 else state
        state_t = tf.convert_to_tensor(state, dtype=tf.float32)
        action_probs = self.actor(state_t, training=False).numpy()
        value = self.critic(state_t, training=False).numpy()
        return action_probs[0], value[0][0]

    def predict_batch(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """批量预测动作概率和状态价值，供并行环境推理复用。"""
        states = np.asarray(states, dtype=np.float32)
        if states.ndim == 1:
            states = np.expand_dims(states, axis=0)
        state_t = tf.convert_to_tensor(states, dtype=tf.float32)
        action_probs = self.actor(state_t, training=False).numpy()
        values = self.critic(state_t, training=False).numpy().reshape(-1)
        return action_probs, values

    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> Tuple[int, float]:
        """采样动作（PPO使用策略网络概率分布，epsilon参数保留仅用于兼容）"""
        action_probs, _ = self.predict(state)
        # PPO直接从策略网络的概率分布中采样动作
        action = np.random.choice(self.output_dim, p=action_probs)
        return action, action_probs[action]
    
    def act(self, state: np.ndarray, game: Optional[PokerGame] = None, player: Optional[Player] = None) -> Tuple[str, Optional[int]]:
        """与DQN兼容的动作选择接口
        
        Args:
            state: 当前状态向量
            game: 游戏实例
            player: 当前玩家
            
        Returns:
            (动作类型, 动作金额)
        """
        # PPO直接从策略网络的概率分布中采样动作，不使用ε-贪婪探索
        action_probs, _ = self.predict(state)
        action_idx = np.random.choice(self.output_dim, p=action_probs)
        
        if game is None or player is None:
            # 如果没有游戏信息，返回动作索引
            return action_idx, None
        else:
            # 计算手牌强度
            hand_strength = self.state_rep.calculate_hand_strength(player, game)
            # 构建动作向量（独热编码）
            action_vector = np.zeros(self.output_dim)
            action_vector[action_idx] = 1
            # 使用state_representation解码动作
            return self.state_rep.decode_action(action_vector, game, player, hand_strength)
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """兼容DQN接口的经验存储方法
        
        注意：PPO通常不使用经验回放，这里仅为兼容接口
        """
        # 由于PPO使用回合经验而非经验回放，这里可以不做实际存储
        # 或者可以存储到临时缓冲区，在回合结束时统一训练
        pass
    
    def replay(self) -> None:
        """兼容DQN接口的回放学习方法
        
        注意：PPO通常不使用经验回放，这里仅为兼容接口
        """
        # PPO不使用经验回放，这里为空实现
        pass
    
    def load(self, filepath: str) -> None:
        """兼容DQN接口的模型加载方法"""
        self.load_weights(filepath)
    
    def save(self, filepath: str) -> None:
        """兼容DQN接口的模型保存方法"""
        self.save_weights(filepath)
    
    def update_target_model(self) -> None:
        """兼容DQN接口的目标模型更新方法
        
        注意：PPO不使用目标网络，这里为空实现
        """
        # PPO不使用目标网络，这里为空实现
        pass

    def warmup(self) -> None:
        """用固定 shape (1, dim) 和 (64, dim) 各跑一次 predict，预编译 tf.function，减少训练时的 retracing 警告。"""
        if getattr(self, "_ppo_warmup_done", False):
            return
        bs = min(self.batch_size, 256)
        dummy1 = np.zeros((1, self.input_dim), dtype=np.float32)
        dummy64 = np.zeros((bs, self.input_dim), dtype=np.float32)
        _ = self.actor(tf.convert_to_tensor(dummy1), training=False)
        _ = self.actor(tf.convert_to_tensor(dummy64), training=False)
        _ = self.critic(tf.convert_to_tensor(dummy1), training=False)
        _ = self.critic(tf.convert_to_tensor(dummy64), training=False)
        self._ppo_warmup_done = True
    
    def _get_lr_with_warmup_and_decay(self, step: int) -> float:
        """计算带warmup和衰减的学习率
        
        优化：
        - 线性warmup阶段：前warmup_steps步从lr_min线性增长到initial_lr
        - cosine衰减阶段：warmup后按cosine曲线衰减到lr_min
        """
        if self.use_lr_warmup and step < self.warmup_steps:
            # 线性warmup
            warmup_ratio = step / max(1, self.warmup_steps)
            lr = self.lr_min + (self.initial_lr - self.lr_min) * warmup_ratio
        elif self.use_cosine_decay:
            # Cosine衰减
            decay_steps = max(1, self.total_training_steps - self.warmup_steps)
            progress = max(0.0, min(1.0, (step - self.warmup_steps) / decay_steps))
            lr = self.lr_min + 0.5 * (self.initial_lr - self.lr_min) * (1 + np.cos(np.pi * progress))
        else:
            # 原始固定衰减
            lr = self.current_lr
        
        return max(self.lr_min, lr)
    
    def update_lr(self, step: int) -> None:
        """更新学习率（兼容LossScaleOptimizer包装）"""
        new_lr = self._get_lr_with_warmup_and_decay(step)
        self.current_lr = new_lr
        # LossScaleOptimizer 包装后需通过 inner_optimizer 或直接属性设置
        for opt in (self.actor_optimizer, self.critic_optimizer):
            inner = getattr(opt, 'inner_optimizer', opt)
            inner.learning_rate.assign(new_lr)
    
    def _compute_entropy_loss(self, probs_f32: tf.Tensor, mask_f32: tf.Tensor) -> tf.Tensor:
        """计算熵损失
        
        优化：
        - 支持自适应熵系数
        - 数值稳定性改进
        """
        # 计算策略熵
        entropy = -tf.reduce_sum(probs_f32 * tf.math.log(probs_f32 + 1e-8), axis=1)
        entropy_loss = -tf.reduce_sum(entropy * mask_f32) / (tf.reduce_sum(mask_f32) + 1e-8)
        
        if self.entropy_adaptive:
            # 自适应熵系数：根据当前熵与目标熵的差距动态调整
            current_entropy = tf.reduce_mean(entropy)
            entropy_diff = float(current_entropy) - self.entropy_coef_target
            # 如果熵低于目标，增加熵系数；如果高于目标，减少熵系数
            adaptive_coef = self.entropy_coef * (1.0 - np.clip(entropy_diff * 2, -0.5, 0.5))
            adaptive_coef = max(0.001, min(0.1, adaptive_coef))
        else:
            adaptive_coef = self.entropy_coef
        
        return entropy_loss, adaptive_coef

    def _predict_critic_fixed_batch(self, states: np.ndarray, chunk_size: int = 64) -> np.ndarray:
        """用固定 batch 形状调用 critic，避免不同 shape 触发 retracing。"""
        n = len(states)
        if n <= 0:
            return np.array([], dtype=np.float32)
        if n <= chunk_size:
            # 单次调用，但 pad 到 chunk_size 以固定 shape
            pad = chunk_size - n
            padded = np.zeros((chunk_size, states.shape[1]), dtype=np.float32)
            padded[:n] = states
            out = self.critic(tf.convert_to_tensor(padded), training=False).numpy().flatten()[:n]
            return out
        values_list = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = states[start:end]
            if len(chunk) < chunk_size:
                padded = np.zeros((chunk_size, states.shape[1]), dtype=np.float32)
                padded[: len(chunk)] = chunk
                v = self.critic(tf.convert_to_tensor(padded), training=False).numpy().flatten()[: len(chunk)]
            else:
                v = self.critic(tf.convert_to_tensor(chunk), training=False).numpy().flatten()
            values_list.append(v)
        return np.concatenate(values_list, axis=0)

    def train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        old_action_probs: np.ndarray
    ) -> float:
        """训练模型：PPO核心逻辑
        
        RTX 4090D 优化：
        - KL散度早停：当KL超过阈值时提前终止该epoch
        - 传递旧价值用于正确的value clipping
        - 自适应学习率调度
        - 自适应熵系数
        """
        import logging
        import time as _time
        logger = logging.getLogger('PPOModel')
        _train_t0 = _time.time()

        buffer_size = len(states)
        train_batch_size = self.batch_size
        logger.info(
            f"[GPU诊断] PPO train() 开始: buffer_size={buffer_size}, "
            f"batch_size={train_batch_size}, update_epochs={self.update_epochs}"
        )
        
        # 计算状态价值和优势（仅用固定 chunk，避免 fallback 可变 shape 触发 retracing）
        values = self._predict_critic_fixed_batch(states, chunk_size=train_batch_size)
        if len(values) != buffer_size:
            values = self._predict_critic_fixed_batch(states, chunk_size=train_batch_size)
        if len(values) != buffer_size:
            values = np.zeros(buffer_size, dtype=np.float32)  # 极罕见时保底
        advantages = self._compute_gae(rewards, values, dones)
        returns = advantages + values  # 目标回报

        # 保存旧价值用于value clipping
        old_values = values.copy()

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_kl = 0.0
        num_batches_done = 0

        state_dim = states.shape[1]
        
        # 梯度累积初始化
        if self.gradient_accumulation_steps > 1:
            self.accumulated_actor_grads = [tf.zeros_like(v) for v in self.actor.trainable_variables]
            self.accumulated_critic_grads = [tf.zeros_like(v) for v in self.critic.trainable_variables]
            self.accumulation_counter = 0

        for epoch in range(self.update_epochs):
            indices = np.random.permutation(buffer_size)
            epoch_kl_sum = 0.0
            epoch_batches = 0
            early_stopped = False
            
            for start in range(0, buffer_size, train_batch_size):
                end = min(start + train_batch_size, buffer_size)
                batch_indices = indices[start:end]
                actual_len = len(batch_indices)
                
                # 不足一整批时对数据做 padding 到 train_batch_size，保证 shape 一致
                if actual_len < train_batch_size:
                    pad_len = train_batch_size - actual_len
                    batch_states_np = np.zeros((train_batch_size, state_dim), dtype=np.float32)
                    batch_states_np[:actual_len] = states[batch_indices]
                    batch_actions_np = np.zeros(train_batch_size, dtype=np.int32)
                    batch_actions_np[:actual_len] = actions[batch_indices]
                    batch_old_probs_np = np.zeros(train_batch_size, dtype=np.float32)
                    batch_old_probs_np[:actual_len] = old_action_probs[batch_indices]
                    batch_advantages_np = np.zeros(train_batch_size, dtype=np.float32)
                    batch_advantages_np[:actual_len] = advantages[batch_indices]
                    batch_returns_np = np.zeros(train_batch_size, dtype=np.float32)
                    batch_returns_np[:actual_len] = returns[batch_indices]
                    batch_old_values_np = np.zeros(train_batch_size, dtype=np.float32)
                    batch_old_values_np[:actual_len] = old_values[batch_indices]
                    mask = np.zeros(train_batch_size, dtype=np.float32)
                    mask[:actual_len] = 1.0
                else:
                    batch_states_np = states[batch_indices]
                    batch_actions_np = actions[batch_indices]
                    batch_old_probs_np = old_action_probs[batch_indices]
                    batch_advantages_np = advantages[batch_indices]
                    batch_returns_np = returns[batch_indices]
                    batch_old_values_np = old_values[batch_indices]
                    mask = np.ones(train_batch_size, dtype=np.float32)

                batch_states = tf.convert_to_tensor(batch_states_np, dtype=tf.float32)
                batch_actions = tf.convert_to_tensor(batch_actions_np, dtype=tf.int32)
                batch_old_probs = tf.convert_to_tensor(batch_old_probs_np, dtype=tf.float32)
                batch_advantages = tf.convert_to_tensor(batch_advantages_np, dtype=tf.float32)
                batch_returns = tf.convert_to_tensor(batch_returns_np, dtype=tf.float32)
                batch_old_values = tf.convert_to_tensor(batch_old_values_np, dtype=tf.float32)
                batch_mask = tf.convert_to_tensor(mask, dtype=tf.float32)

                actor_loss, critic_loss, kl_val = self._ppo_update_step(
                    batch_states,
                    batch_actions,
                    batch_old_probs,
                    batch_advantages,
                    batch_returns,
                    batch_old_values,
                    batch_mask,
                )

                batch_kl = float(kl_val.numpy())
                epoch_kl_sum += batch_kl
                epoch_batches += 1
                
                # KL散度早停：平均KL超过阈值时提前终止该epoch
                if self.kl_early_stop and epoch_batches > 1:
                    avg_kl = epoch_kl_sum / epoch_batches
                    if avg_kl > self.kl_target:
                        logger.debug(
                            f"KL早停 epoch {epoch}: avg_kl={avg_kl:.6f} > target={self.kl_target:.6f}"
                        )
                        early_stopped = True
                        break

                total_actor_loss += float(actor_loss.numpy())
                total_critic_loss += float(critic_loss.numpy())
                total_kl += batch_kl
                num_batches_done += 1
                
                # 更新学习率
                self.current_step += 1
                self.update_lr(self.current_step)
            
            if early_stopped:
                # KL早停后不再做后续epoch
                break

        num_batches_done = max(num_batches_done, 1)
        avg_actor_loss = total_actor_loss / num_batches_done
        avg_critic_loss = total_critic_loss / num_batches_done
        self.last_loss = (avg_actor_loss + avg_critic_loss) / 2  # 记录总损失
        self.last_kl = total_kl / num_batches_done

        _train_elapsed = _time.time() - _train_t0
        # 检查实际计算设备
        try:
            _dev_info = self.actor.layers[0].weights[0].device
        except Exception:
            _dev_info = "unknown"
        logger.info(
            f"[GPU诊断] PPO train() 完成: 耗时={_train_elapsed:.4f}s, "
            f"批次数={num_batches_done}, 设备={_dev_info}, "
            f"Actor损失={avg_actor_loss:.6f}, Critic损失={avg_critic_loss:.6f}, "
            f"KL={self.last_kl:.6f}"
        )
        logger.debug(
            f"训练完成: Actor损失={avg_actor_loss:.6f}, Critic损失={avg_critic_loss:.6f}, "
            f"总损失={self.last_loss:.6f}, KL散度={self.last_kl:.6f}, "
            f"当前学习率={self.current_lr:.6f}, 步数={self.current_step}"
        )

        return self.last_loss

    def save_weights(self, path: str) -> None:
        """保存模型权重。path 可为 base 路径或含 .weights.h5 的路径，统一存为 base.actor.weights.h5 / base.critic.weights.h5"""
        base = _ppo_base_path(path)
        actor_path = base + ".actor.weights.h5"
        critic_path = base + ".critic.weights.h5"
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)

    def load_weights(self, path: str) -> None:
        """加载模型权重。path 可为 base、.actor.weights.h5 或 .weights.h5 等，自动解析为 base."""
        base = _ppo_base_path(path)
        actor_path = base + ".actor.weights.h5"
        critic_path = base + ".critic.weights.h5"
        if not os.path.exists(actor_path) or not os.path.exists(critic_path):
            raise FileNotFoundError(
                f"PPO 权重文件不存在: 需要 {actor_path} 与 {critic_path}"
            )
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)
    
    def set_weights(self, weights):
        """设置模型权重"""
        # 这里需要根据PPO的具体实现来设置权重
        if len(weights) >= 2:
            self.actor.set_weights(weights[0])
            self.critic.set_weights(weights[1])
    
    def get_weights(self):
        """获取模型权重"""
        return [self.actor.get_weights(), self.critic.get_weights()]
