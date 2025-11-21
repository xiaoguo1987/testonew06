import tensorflow as tf
import numpy as np
from typing import List, Tuple, Optional, Dict
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from ..game.poker_game import Player, PokerGame
from .state_representation import StateRepresentation

class PPOModel:
    """近端策略优化（PPO）模型，兼容德州扑克AI训练框架"""
    def __init__(
        self,
        input_dim: int = 18,  # 状态输入维度（与DQN/PG一致）
        output_dim: int = 5,  # 动作维度（与DQN保持一致）
        lr: float = 3e-4,
        clip_epsilon: float = 0.2,  # PPO Clip参数
        gamma: float = 0.99,  # 折扣因子
        lambda_: float = 0.95,  # GAE参数
        update_epochs: int = 10,  # 每批数据更新轮次
        batch_size: int = 64,  # 批次大小
        epsilon: float = 0.1,  # 探索率（兼容DQN接口）
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lambda_ = lambda_
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.last_loss = 0.0  # 记录最新损失（用于TensorBoard）
        self.epsilon = epsilon  # 探索率
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.state_rep = StateRepresentation()  # 状态表示器

        # 构建策略网络（Actor）和价值网络（Critic）
        self.actor = self._build_actor()
        self.critic = self._build_critic()

        # 优化器
        self.actor_optimizer = Adam(learning_rate=lr)
        self.critic_optimizer = Adam(learning_rate=lr)

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
        action_probs = self.actor.predict(state, verbose=0)
        value = self.critic.predict(state, verbose=0)
        return action_probs[0], value[0][0]

    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> Tuple[int, float]:
        """采样动作（含探索机制）"""
        action_probs, _ = self.predict(state)
        # epsilon-贪婪探索
        if np.random.random() < epsilon:
            action = np.random.choice(self.output_dim)
        else:
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
        # 探索与利用
        if np.random.rand() <= self.epsilon:
            # 随机选择动作
            action_idx = np.random.randint(0, self.output_dim)
        else:
            # 使用PPO模型选择动作
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

    def train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        old_action_probs: np.ndarray
    ) -> float:
        """训练模型：PPO核心逻辑"""
        # 计算状态价值和优势
        values = self.critic.predict(states, verbose=0).flatten()
        advantages = self._compute_gae(rewards, values, dones)
        returns = advantages + values  # 目标回报

        # 转换为TensorFlow张量
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        old_action_probs = tf.convert_to_tensor(old_action_probs, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        # 多轮更新策略
        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for _ in range(self.update_epochs):
            # 随机打乱数据（分批训练）
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # 批次数据
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_probs = old_action_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # 更新策略网络（Actor）
                with tf.GradientTape() as tape:
                    batch_action_probs = self.actor(batch_states)
                    # 提取选中动作的概率
                    action_masks = tf.one_hot(batch_actions, self.output_dim)
                    batch_current_probs = tf.reduce_sum(
                        batch_action_probs * action_masks, axis=1
                    )
                    # PPO Clip损失
                    ratio = batch_current_probs / (batch_old_probs + 1e-8)  # 避免除零
                    clipped_ratio = tf.clip_by_value(
                        ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                    )
                    actor_loss = -tf.reduce_mean(
                        tf.minimum(ratio * batch_advantages, clipped_ratio * batch_advantages)
                    )

                actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(
                    zip(actor_grads, self.actor.trainable_variables)
                )

                # 更新价值网络（Critic）
                with tf.GradientTape() as tape:
                    batch_values = self.critic(batch_states)
                    critic_loss = tf.reduce_mean(tf.square(batch_values - batch_returns[:, None]))

                critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                self.critic_optimizer.apply_gradients(
                    zip(critic_grads, self.critic.trainable_variables)
                )

                total_actor_loss += actor_loss.numpy()
                total_critic_loss += critic_loss.numpy()

        # 计算平均损失
        avg_actor_loss = total_actor_loss / (self.update_epochs * (len(states) // self.batch_size + 1))
        avg_critic_loss = total_critic_loss / (self.update_epochs * (len(states) // self.batch_size + 1))
        self.last_loss = (avg_actor_loss + avg_critic_loss) / 2  # 记录总损失

        return self.last_loss

    def save_weights(self, path: str) -> None:
        """保存模型权重（兼容现有checkpoint机制）"""
        # 分别保存Actor和Critic权重
        actor_path = path.replace(".weights.h5", ".actor.weights.h5")
        critic_path = path.replace(".weights.h5", ".critic.weights.h5")
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)

    def load_weights(self, path: str) -> None:
        """加载模型权重（兼容现有checkpoint机制）"""
        try:
            actor_path = path.replace(".weights.h5", ".actor.weights.h5")
            critic_path = path.replace(".weights.h5", ".critic.weights.h5")
            self.actor.load_weights(actor_path)
            self.critic.load_weights(critic_path)
        except Exception as e:
            raise ValueError(f"PPO模型权重加载失败：{str(e)}")