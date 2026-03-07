import tensorflow as tf
import numpy as np
import os
from typing import List, Tuple, Optional, Dict
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from ..game.poker_game import Player, PokerGame
from .state_representation import StateRepresentation
from ..config.ppo_config import PPOConfig, DEFAULT_PPO_CONFIG


def _ppo_base_path(path: str) -> str:
    """将任意 PPO 相关路径规范为 base 路径（无后缀），用于 actor/critic 文件名."""
    path = path.replace(".actor.weights.h5", "").replace(".critic.weights.h5", "")
    path = path.replace(".weights.h5", "").rstrip("/")
    return path


class PPOModel:
    """近端策略优化（PPO）模型，兼容德州扑克AI训练框架"""
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

        # 构建策略网络（Actor）和价值网络（Critic）
        self.actor = self._build_actor()
        self.critic = self._build_critic()

        # 优化器
        self.actor_optimizer = Adam(learning_rate=lr_val)
        self.critic_optimizer = Adam(learning_rate=lr_val)

    @tf.function(reduce_retracing=True)
    def _ppo_update_step(
        self,
        batch_states: tf.Tensor,
        batch_actions: tf.Tensor,
        batch_old_probs: tf.Tensor,
        batch_advantages: tf.Tensor,
        batch_returns: tf.Tensor,
        batch_mask: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """单个 PPO mini-batch 更新（编译后执行，降低 Python 调度开销）。"""
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
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            per_sample = tf.minimum(ratio * advantages_f32, clipped_ratio * advantages_f32)
            policy_loss = -tf.reduce_sum(per_sample * mask_f32) / (tf.reduce_sum(mask_f32) + 1e-8)

            # 熵正则：抑制策略过早塌缩
            entropy = -tf.reduce_sum(probs_f32 * tf.math.log(probs_f32 + 1e-8), axis=1)
            entropy_loss = -tf.reduce_sum(entropy * mask_f32) / (tf.reduce_sum(mask_f32) + 1e-8)
            actor_loss = policy_loss + self.entropy_coef * entropy_loss

        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        actor_grads = [
            tf.clip_by_norm(grad, self.max_grad_norm) if grad is not None else None
            for grad in actor_grads
        ]
        self.actor_optimizer.apply_gradients(
            [(g, v) for g, v in zip(actor_grads, self.actor.trainable_variables) if g is not None]
        )

        # 更新价值网络（Critic）
        with tf.GradientTape() as critic_tape:
            batch_values = self.critic(batch_states, training=True)
            values_f32 = tf.cast(batch_values, tf.float32)
            returns_f32 = tf.cast(batch_returns, tf.float32)
            mask_f32 = tf.cast(batch_mask, tf.float32)
            sq = tf.square(values_f32 - returns_f32[:, None])
            critic_loss = tf.reduce_sum(sq * mask_f32[:, None]) / (tf.reduce_sum(mask_f32) + 1e-8)

        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        critic_grads = [
            tf.clip_by_norm(grad, self.max_grad_norm) if grad is not None else None
            for grad in critic_grads
        ]
        self.critic_optimizer.apply_gradients(
            [(g, v) for g, v in zip(critic_grads, self.critic.trainable_variables) if g is not None]
        )

        return actor_loss, critic_loss

    def _build_actor(self) -> models.Model:
        """构建策略网络：输出动作概率分布"""
        inputs = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(64, activation="relu")(inputs)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(32, activation="relu")(x)
        outputs = layers.Dense(self.output_dim, activation="softmax")(x)  # 动作概率
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    def _build_critic(self) -> models.Model:
        """构建价值网络：估计状态价值"""
        inputs = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(64, activation="relu")(inputs)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(32, activation="relu")(x)
        outputs = layers.Dense(1, activation="linear")(x)  # 状态价值
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    def _compute_gae(
        self, rewards: List[float], values: List[float], dones: List[bool]
    ) -> np.ndarray:
        """计算广义优势估计（GAE）"""
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_advantage = 0.0
        last_value = 0.0

        # 逆序计算优势
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_advantage = delta
            else:
                delta = rewards[t] + self.gamma * last_value - values[t]
                last_advantage = delta + self.gamma * self.lambda_ * last_advantage
            advantages[t] = last_advantage
            last_value = values[t]

        # 优势标准化（提升稳定性）
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return advantages

    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测动作概率和状态价值（兼容Agent接口）"""
        state = np.expand_dims(state, axis=0) if len(state.shape) == 1 else state
        state_t = tf.convert_to_tensor(state, dtype=tf.float32)
        action_probs = self.actor(state_t, training=False).numpy()
        value = self.critic(state_t, training=False).numpy()
        return action_probs[0], value[0][0]

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
        """训练模型：PPO核心逻辑。使用固定 batch shape 减少 tf.function retracing。"""
        import logging
        logger = logging.getLogger('PPOModel')

        buffer_size = len(states)
        train_batch_size = self.batch_size
        # 计算状态价值和优势（仅用固定 chunk，避免 fallback 可变 shape 触发 retracing）
        values = self._predict_critic_fixed_batch(states, chunk_size=train_batch_size)
        if len(values) != buffer_size:
            values = self._predict_critic_fixed_batch(states, chunk_size=train_batch_size)
        if len(values) != buffer_size:
            values = np.zeros(buffer_size, dtype=np.float32)  # 极罕见时保底
        advantages = self._compute_gae(rewards, values, dones)
        returns = advantages + values  # 目标回报

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        num_batches_done = 0

        state_dim = states.shape[1]
        for epoch in range(self.update_epochs):
            indices = np.random.permutation(buffer_size)
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
                    mask = np.zeros(train_batch_size, dtype=np.float32)
                    mask[:actual_len] = 1.0
                else:
                    batch_states_np = states[batch_indices]
                    batch_actions_np = actions[batch_indices]
                    batch_old_probs_np = old_action_probs[batch_indices]
                    batch_advantages_np = advantages[batch_indices]
                    batch_returns_np = returns[batch_indices]
                    mask = np.ones(train_batch_size, dtype=np.float32)

                batch_states = tf.convert_to_tensor(batch_states_np, dtype=tf.float32)
                batch_actions = tf.convert_to_tensor(batch_actions_np, dtype=tf.int32)
                batch_old_probs = tf.convert_to_tensor(batch_old_probs_np, dtype=tf.float32)
                batch_advantages = tf.convert_to_tensor(batch_advantages_np, dtype=tf.float32)
                batch_returns = tf.convert_to_tensor(batch_returns_np, dtype=tf.float32)
                batch_mask = tf.convert_to_tensor(mask, dtype=tf.float32)

                actor_loss, critic_loss = self._ppo_update_step(
                    batch_states,
                    batch_actions,
                    batch_old_probs,
                    batch_advantages,
                    batch_returns,
                    batch_mask,
                )

                total_actor_loss += float(actor_loss.numpy())
                total_critic_loss += float(critic_loss.numpy())
                num_batches_done += 1

        num_batches_done = max(num_batches_done, 1)
        avg_actor_loss = total_actor_loss / num_batches_done
        avg_critic_loss = total_critic_loss / num_batches_done
        self.last_loss = (avg_actor_loss + avg_critic_loss) / 2  # 记录总损失

        # 学习率衰减
        self.current_lr = max(self.lr_min, self.current_lr * self.lr_decay_rate)
        # 更新优化器学习率
        self.actor_optimizer.learning_rate.assign(self.current_lr)
        self.critic_optimizer.learning_rate.assign(self.current_lr)
        
        logger.debug(f"训练完成: Actor损失={avg_actor_loss:.6f}, Critic损失={avg_critic_loss:.6f}, 总损失={self.last_loss:.6f}, 当前学习率={self.current_lr:.6f}")

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
