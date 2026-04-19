"""大参数量 PPO 模型 — SEED RL GPU-Centric 架构专用

网络架构可配置（默认 2048-2048-1024-512，约 14M 参数）：
- 充分压榨 RTX 4090D 的 82 TFLOPS FP32 算力
- 支持混合精度 (mixed_float16) + XLA JIT
- 复用现有 PPO 算法（clip, GAE, KL 早停, 价值裁剪）
- predict_batch() 供集中式 GPU 推理服务器调用
"""

import tensorflow as tf
import numpy as np
import os
import logging
import time as _time
from typing import List, Tuple, Optional
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from ..config.ppo_config import PPOConfig, DEFAULT_PPO_CONFIG

logger = logging.getLogger('LargePPOModel')


def _ppo_base_path(path: str) -> str:
    path = path.replace(".actor.weights.h5", "").replace(".critic.weights.h5", "")
    path = path.replace(".weights.h5", "").rstrip("/")
    return path


class LargePPOModel:
    """大参数量 PPO 模型（Actor-Critic），支持可配置隐藏层维度。

    默认架构（每个网络 ~7.1M 参数，总计 ~14.2M）：
        134 → 2048 → 2048(残差) → 1024 → 512 → output

    与 PPOModel 的主要区别：
    1. hidden_dims 参数化，默认 [2048, 2048, 1024, 512]
    2. 残差连接位置可配（residual_layer_idx）
    3. 更大 batch_size 默认值（2048）
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: int = 5,
        hidden_dims: Optional[List[int]] = None,
        residual_layer_idx: int = 1,
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
        self.hidden_dims = hidden_dims or getattr(cfg, 'hidden_dims', [2048, 2048, 1024, 512])
        self.residual_layer_idx = residual_layer_idx

        self.clip_epsilon = clip_epsilon if clip_epsilon is not None else cfg.clip_epsilon
        self.gamma = gamma if gamma is not None else cfg.gamma
        self.lambda_ = lambda_ if lambda_ is not None else cfg.lambda_
        self.entropy_coef = entropy_coef if entropy_coef is not None else getattr(cfg, 'entropy_coef', 0.02)
        self.update_epochs = update_epochs if update_epochs is not None else cfg.update_epochs
        self.batch_size = max(1, int(batch_size if batch_size is not None else cfg.batch_size))
        self.max_grad_norm = getattr(cfg, 'max_grad_norm', 0.5)
        self.last_loss = 0.0

        lr_val = lr if lr is not None else cfg.lr
        self.initial_lr = lr_val
        self.current_lr = lr_val
        self.lr_min = getattr(cfg, 'lr_min', 1e-6)

        # 学习率调度
        self.use_lr_warmup = getattr(cfg, 'use_lr_warmup', True)
        self.warmup_steps = getattr(cfg, 'warmup_steps', 3000)
        self.use_cosine_decay = getattr(cfg, 'use_cosine_decay', True)
        self.total_training_steps = getattr(cfg, 'total_training_steps', 300000)
        self.current_step = 0

        # 自适应熵系数
        self.entropy_coef_target = getattr(cfg, 'entropy_coef_target', 0.01)
        self.entropy_adaptive = getattr(cfg, 'entropy_adaptive', True)

        # 价值损失裁剪
        self.value_clip = getattr(cfg, 'value_clip', 0.2)

        # KL 早停
        self.kl_target = getattr(cfg, 'kl_target', 0.015)
        self.kl_early_stop = getattr(cfg, 'kl_early_stop', True)
        self.last_kl = 0.0

        # 构建网络
        self.actor = self._build_network(softmax_output=True)
        self.critic = self._build_network(softmax_output=False)

        # 优化器
        base_actor_opt = Adam(learning_rate=lr_val)
        base_critic_opt = Adam(learning_rate=lr_val)
        self._use_mixed_precision = tf.keras.mixed_precision.global_policy().name != 'float32'
        if self._use_mixed_precision:
            self.actor_optimizer = LossScaleOptimizer(base_actor_opt)
            self.critic_optimizer = LossScaleOptimizer(base_critic_opt)
        else:
            self.actor_optimizer = base_actor_opt
            self.critic_optimizer = base_critic_opt

        param_count = self.actor.count_params() + self.critic.count_params()
        logger.info(
            f"LargePPOModel 初始化: hidden_dims={self.hidden_dims}, "
            f"residual@layer{self.residual_layer_idx}, "
            f"总参数量={param_count:,} ({param_count/1e6:.1f}M)"
        )

    # ------------------------------------------------------------------
    #  网络构建
    # ------------------------------------------------------------------

    def _build_network(self, softmax_output: bool = True) -> models.Model:
        """构建参数化的 Dense + LayerNorm + Residual 网络。"""
        inputs = layers.Input(shape=(self.input_dim,))
        x = inputs
        prev_dim = self.input_dim

        for i, hdim in enumerate(self.hidden_dims):
            dense_out = layers.Dense(hdim, kernel_initializer='he_uniform')(x)
            dense_out = layers.LayerNormalization()(dense_out)

            if i == self.residual_layer_idx and prev_dim == hdim:
                # 残差连接（需要维度匹配）
                x = layers.add([x, dense_out])
            else:
                x = dense_out

            x = layers.Activation('relu')(x)
            prev_dim = hdim

        if softmax_output:
            outputs = layers.Dense(
                self.output_dim, activation='softmax',
                kernel_initializer='glorot_uniform'
            )(x)
        else:
            outputs = layers.Dense(
                1, activation='linear',
                kernel_initializer='glorot_uniform'
            )(x)

        return models.Model(inputs=inputs, outputs=outputs)

    # ------------------------------------------------------------------
    #  推理
    # ------------------------------------------------------------------

    @tf.function(reduce_retracing=True)
    def _predict_actor(self, states):
        return self.actor(states, training=False)

    @tf.function(reduce_retracing=True)
    def _predict_critic(self, states):
        return self.critic(states, training=False)

    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        state = np.expand_dims(state, axis=0) if state.ndim == 1 else state
        st = tf.convert_to_tensor(state, dtype=tf.float32)
        ap = self._predict_actor(st).numpy()
        v = self._predict_critic(st).numpy()
        return ap[0], float(v[0, 0])

    def predict_batch(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """批量推理 — GPU 集中式推理服务器核心接口。
        
        使用 @tf.function 编译的内部方法，消除 Python 调用开销。
        """
        states = np.asarray(states, dtype=np.float32)
        if states.ndim == 1:
            states = np.expand_dims(states, axis=0)
        st = tf.convert_to_tensor(states, dtype=tf.float32)
        ap = self._predict_actor(st).numpy()
        v = self._predict_critic(st).numpy().reshape(-1)
        return ap, v

    # ------------------------------------------------------------------
    #  GAE
    # ------------------------------------------------------------------

    def _compute_gae(
        self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray
    ) -> np.ndarray:
        n = len(rewards)
        if n == 0:
            return np.array([], dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        next_values = np.zeros_like(values)
        next_values[:-1] = values[1:]
        deltas = rewards + self.gamma * next_values * (1.0 - dones) - values

        advantages = np.zeros_like(deltas)
        advantages[-1] = deltas[-1]
        for t in range(n - 2, -1, -1):
            advantages[t] = deltas[t] + self.gamma * self.lambda_ * (1.0 - dones[t]) * advantages[t + 1]

        std = np.std(advantages)
        if std > 1e-8:
            advantages = (advantages - np.mean(advantages)) / (std + 1e-8)
        else:
            advantages = advantages - np.mean(advantages)
        return np.clip(advantages, -10.0, 10.0)

    # ------------------------------------------------------------------
    #  PPO 训练核心
    # ------------------------------------------------------------------

    @tf.function(reduce_retracing=True)
    def _ppo_update_step(
        self,
        batch_states, batch_actions, batch_old_probs,
        batch_advantages, batch_returns, batch_old_values, batch_mask,
    ):
        # Actor
        with tf.GradientTape() as actor_tape:
            probs = tf.cast(self.actor(batch_states, training=True), tf.float32)
            action_masks = tf.cast(tf.one_hot(batch_actions, self.output_dim), tf.float32)
            old_p = tf.cast(batch_old_probs, tf.float32)
            adv = tf.cast(batch_advantages, tf.float32)
            msk = tf.cast(batch_mask, tf.float32)

            cur_p = tf.reduce_sum(probs * action_masks, axis=1)
            ratio = cur_p / (old_p + 1e-8)

            log_ratio = tf.math.log(cur_p + 1e-8) - tf.math.log(old_p + 1e-8)
            mean_kl = tf.reduce_mean((tf.exp(log_ratio) - 1.0) - log_ratio)

            clipped = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            surr = tf.minimum(ratio * adv, clipped * adv)
            policy_loss = -tf.reduce_sum(surr * msk) / (tf.reduce_sum(msk) + 1e-8)

            entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1)
            entropy_loss = -tf.reduce_sum(entropy * msk) / (tf.reduce_sum(msk) + 1e-8)
            actor_loss = policy_loss + self.entropy_coef * entropy_loss

        a_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        a_grads, _ = tf.clip_by_global_norm(a_grads, self.max_grad_norm)
        self.actor_optimizer.apply_gradients(
            [(g, v) for g, v in zip(a_grads, self.actor.trainable_variables) if g is not None]
        )

        # Critic
        with tf.GradientTape() as critic_tape:
            vals = tf.cast(self.critic(batch_states, training=True), tf.float32)
            old_v = tf.cast(batch_old_values, tf.float32)
            rets = tf.cast(batch_returns, tf.float32)
            msk = tf.cast(batch_mask, tf.float32)

            vals_clipped = old_v[:, None] + tf.clip_by_value(
                vals - old_v[:, None], -self.value_clip, self.value_clip
            )
            vf1 = tf.square(vals - rets[:, None])
            vf2 = tf.square(vals_clipped - rets[:, None])
            critic_loss = tf.reduce_sum(
                tf.maximum(vf1, vf2)[:, 0] * msk
            ) / (tf.reduce_sum(msk) + 1e-8)

        c_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        c_grads, _ = tf.clip_by_global_norm(c_grads, self.max_grad_norm)
        self.critic_optimizer.apply_gradients(
            [(g, v) for g, v in zip(c_grads, self.critic.trainable_variables) if g is not None]
        )

        return actor_loss, critic_loss, mean_kl

    def _predict_critic_batch(self, states: np.ndarray, chunk_size: int = 256) -> np.ndarray:
        n = len(states)
        if n <= 0:
            return np.array([], dtype=np.float32)
        if n <= chunk_size:
            pad = chunk_size - n
            padded = np.zeros((chunk_size, states.shape[1]), dtype=np.float32)
            padded[:n] = states
            return self.critic(tf.convert_to_tensor(padded), training=False).numpy().flatten()[:n]
        parts = []
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            chunk = states[s:e]
            if len(chunk) < chunk_size:
                padded = np.zeros((chunk_size, states.shape[1]), dtype=np.float32)
                padded[:len(chunk)] = chunk
                parts.append(self.critic(tf.convert_to_tensor(padded), training=False).numpy().flatten()[:len(chunk)])
            else:
                parts.append(self.critic(tf.convert_to_tensor(chunk), training=False).numpy().flatten())
        return np.concatenate(parts)

    def train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        old_action_probs: np.ndarray,
    ) -> float:
        t0 = _time.time()
        buf = len(states)
        bs = self.batch_size
        logger.info(f"[LargePPO] train() 开始: buffer={buf}, batch={bs}, epochs={self.update_epochs}")

        values = self._predict_critic_batch(states, chunk_size=bs)
        if len(values) != buf:
            values = np.zeros(buf, dtype=np.float32)
        advantages = self._compute_gae(rewards, values, dones)
        returns = advantages + values
        old_values = values.copy()

        total_a, total_c, total_kl = 0.0, 0.0, 0.0
        done_batches = 0
        sdim = states.shape[1]

        for epoch in range(self.update_epochs):
            idx = np.random.permutation(buf)
            epoch_kl, epoch_n = 0.0, 0
            stopped = False

            for start in range(0, buf, bs):
                end = min(start + bs, buf)
                bi = idx[start:end]
                alen = len(bi)

                if alen < bs:
                    s_np = np.zeros((bs, sdim), dtype=np.float32); s_np[:alen] = states[bi]
                    a_np = np.zeros(bs, dtype=np.int32); a_np[:alen] = actions[bi]
                    op_np = np.zeros(bs, dtype=np.float32); op_np[:alen] = old_action_probs[bi]
                    ad_np = np.zeros(bs, dtype=np.float32); ad_np[:alen] = advantages[bi]
                    r_np = np.zeros(bs, dtype=np.float32); r_np[:alen] = returns[bi]
                    ov_np = np.zeros(bs, dtype=np.float32); ov_np[:alen] = old_values[bi]
                    m_np = np.zeros(bs, dtype=np.float32); m_np[:alen] = 1.0
                else:
                    s_np = states[bi]; a_np = actions[bi]; op_np = old_action_probs[bi]
                    ad_np = advantages[bi]; r_np = returns[bi]; ov_np = old_values[bi]
                    m_np = np.ones(bs, dtype=np.float32)

                al, cl, kl = self._ppo_update_step(
                    tf.convert_to_tensor(s_np, tf.float32),
                    tf.convert_to_tensor(a_np, tf.int32),
                    tf.convert_to_tensor(op_np, tf.float32),
                    tf.convert_to_tensor(ad_np, tf.float32),
                    tf.convert_to_tensor(r_np, tf.float32),
                    tf.convert_to_tensor(ov_np, tf.float32),
                    tf.convert_to_tensor(m_np, tf.float32),
                )

                bkl = float(kl.numpy())
                epoch_kl += bkl
                epoch_n += 1

                if self.kl_early_stop and epoch_n >= 1 and (epoch_kl / epoch_n) > self.kl_target:
                    stopped = True
                    break

                total_a += float(al.numpy())
                total_c += float(cl.numpy())
                total_kl += bkl
                done_batches += 1

                self.current_step += 1
                self.update_lr(self.current_step)

            if stopped:
                break

        done_batches = max(done_batches, 1)
        avg_a = total_a / done_batches
        avg_c = total_c / done_batches
        self.last_loss = (avg_a + avg_c) / 2
        self.last_kl = total_kl / done_batches

        elapsed = _time.time() - t0
        logger.info(
            f"[LargePPO] train() 完成: {elapsed:.3f}s, batches={done_batches}, "
            f"actor_loss={avg_a:.6f}, critic_loss={avg_c:.6f}, KL={self.last_kl:.6f}"
        )
        return self.last_loss

    # ------------------------------------------------------------------
    #  学习率调度
    # ------------------------------------------------------------------

    def _get_lr(self, step: int) -> float:
        if self.use_lr_warmup and step < self.warmup_steps:
            r = step / max(1, self.warmup_steps)
            return self.lr_min + (self.initial_lr - self.lr_min) * r
        if self.use_cosine_decay:
            decay = max(1, self.total_training_steps - self.warmup_steps)
            prog = max(0.0, min(1.0, (step - self.warmup_steps) / decay))
            return self.lr_min + 0.5 * (self.initial_lr - self.lr_min) * (1 + np.cos(np.pi * prog))
        return self.current_lr

    def update_lr(self, step: int) -> None:
        new_lr = max(self.lr_min, self._get_lr(step))
        self.current_lr = new_lr
        for opt in (self.actor_optimizer, self.critic_optimizer):
            inner = getattr(opt, 'inner_optimizer', opt)
            inner.learning_rate.assign(new_lr)

    # ------------------------------------------------------------------
    #  预热
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        if getattr(self, '_warmup_done', False):
            return
        bs = min(self.batch_size, 256)
        d1 = np.zeros((1, self.input_dim), dtype=np.float32)
        db = np.zeros((bs, self.input_dim), dtype=np.float32)
        for m in (self.actor, self.critic):
            m(tf.convert_to_tensor(d1), training=False)
            m(tf.convert_to_tensor(db), training=False)
        self._warmup_done = True
        logger.info(f"LargePPOModel warmup 完成 (batch shapes: 1, {bs})")

    # ------------------------------------------------------------------
    #  权重管理
    # ------------------------------------------------------------------

    def get_weights(self):
        return [self.actor.get_weights(), self.critic.get_weights()]

    def set_weights(self, weights):
        if len(weights) >= 2:
            self.actor.set_weights(weights[0])
            self.critic.set_weights(weights[1])

    def get_weights_snapshot(self):
        import copy
        return copy.deepcopy([self.actor.get_weights(), self.critic.get_weights()])

    def apply_weights_snapshot(self, snapshot):
        if len(snapshot) >= 2:
            self.actor.set_weights(snapshot[0])
            self.critic.set_weights(snapshot[1])

    def save_weights(self, path: str) -> None:
        base = _ppo_base_path(path)
        self.actor.save_weights(base + ".actor.weights.h5")
        self.critic.save_weights(base + ".critic.weights.h5")

    def load_weights(self, path: str) -> None:
        base = _ppo_base_path(path)
        ap = base + ".actor.weights.h5"
        cp = base + ".critic.weights.h5"
        if not os.path.exists(ap) or not os.path.exists(cp):
            raise FileNotFoundError(f"权重文件不存在: {ap}, {cp}")
        self.actor.load_weights(ap)
        self.critic.load_weights(cp)

    # ------------------------------------------------------------------
    #  兼容旧接口
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        self.save_weights(path)

    def load(self, path: str) -> None:
        self.load_weights(path)
