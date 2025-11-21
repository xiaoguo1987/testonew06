# 德州扑克DQN代理模型

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from typing import List, Tuple, Dict, Optional
from .state_representation import StateRepresentation
from ..game.poker_game import Player, PokerGame

class DQNAgent:
    """德州扑克DQN智能代理"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int = 5,
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,  # 折扣因子
                 epsilon: float = 1.0,  # 探索率
                 epsilon_min: float = 0.1,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 64,
                 target_update_freq: int = 10,
                 optimizer: str = 'adam'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.optimizer_type = optimizer
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=memory_size)
        
        # 创建主网络和目标网络
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()  # 初始化时复制权重
        
        # 状态表示转换器
        self.state_rep = StateRepresentation()
        
        # 记录训练步数
        self.training_steps = 0
    
    def _build_model(self) -> Model:
        """构建神经网络模型"""
        model = Sequential([
            Input(shape=(self.state_dim,)),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(self.action_dim, activation='linear')  # 输出Q值
        ])
        
        # 根据optimizer_type选择优化器
        if self.optimizer_type.lower() == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer_type.lower() == 'adamax':
            optimizer = tf.keras.optimizers.Adamax(learning_rate=self.learning_rate)
        elif self.optimizer_type.lower() == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        elif self.optimizer_type.lower() == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        else:
            # 默认使用Adam优化器
            logger.warning(f"未知的优化器类型: {self.optimizer_type}，默认使用Adam")
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def update_target_model(self) -> None:
        """更新目标网络权重"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """存储经验到回放缓冲区"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, game: Optional[PokerGame] = None, player: Optional[Player] = None) -> Tuple[str, Optional[int]]:
        """根据当前状态选择动作"""
        # 探索与利用
        if np.random.rand() <= self.epsilon:
            # 随机选择动作
            action_idx = random.randint(0, self.action_dim - 1)
            if game is None or player is None:
                # 如果没有游戏信息，返回简单动作索引
                return action_idx, None
            else:
                # 解码为具体动作
                action_vector = np.zeros(self.action_dim)
                action_vector[action_idx] = 1
                return self.state_rep.decode_action(action_vector, game, player)
        
        # 利用模型选择最优动作
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        action_idx = np.argmax(q_values)
        
        if game is None or player is None:
            return action_idx, None
        else:
            action_vector = np.zeros(self.action_dim)
            action_vector[action_idx] = 1
            return self.state_rep.decode_action(action_vector, game, player)
    
    def replay(self) -> None:
        """从经验回放中学习"""
        if len(self.memory) < self.batch_size:
            return
        
        # 从记忆中随机采样批次
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([exp[0] for exp in minibatch])
        actions = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch])
        
        # 计算目标Q值
        target_q = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)
        
        for i in range(self.batch_size):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                target_q[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
        
        # 训练模型
        self.model.fit(states, target_q, epochs=1, verbose=0, batch_size=self.batch_size)
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 更新目标网络
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.update_target_model()
    
    def load(self, name: str) -> None:
        """加载模型权重"""
        self.model.load_weights(name)
        self.update_target_model()
    
    def save(self, name: str) -> None:
        """保存模型权重"""
        self.model.save_weights(name)
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """获取状态的Q值"""
        return self.model.predict(state.reshape(1, -1), verbose=0)[0]

# 策略梯度代理 (Policy Gradient Agent)
class PolicyGradientAgent:
    """基于策略梯度的德州扑克代理"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int = 5,
                 learning_rate: float = 0.001,
                 gamma: float = 0.95):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # 创建策略网络
        self.policy_network = self._build_policy_network()
        
        # 状态表示转换器
        self.state_rep = StateRepresentation()
        
        # 记录每个回合的轨迹
        self.states = []
        self.actions = []
        self.rewards = []
    
    def _build_policy_network(self) -> Model:
        """构建策略网络"""
        model = Sequential([
            Input(shape=(self.state_dim,)),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(self.action_dim, activation='softmax')  # 输出动作概率
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy'
        )
        
        return model
    
    def act(self, state: np.ndarray, game: Optional[PokerGame] = None, player: Optional[Player] = None) -> Tuple[str, Optional[int]]:
        """根据当前状态选择动作"""
        # 预测动作概率
        action_probs = self.policy_network.predict(state.reshape(1, -1), verbose=0)[0]
        
        # 基于概率采样动作
        action_idx = np.random.choice(self.action_dim, p=action_probs)
        
        # 记录状态和动作
        self.states.append(state)
        self.actions.append(action_idx)
        
        if game is None or player is None:
            return action_idx, None
        else:
            action_vector = np.zeros(self.action_dim)
            action_vector[action_idx] = 1
            return self.state_rep.decode_action(action_vector, game, player)
    
    def discount_rewards(self, rewards: List[float]) -> np.ndarray:
        """计算折扣奖励"""
        discounted = np.zeros_like(rewards, dtype=float)
        running_add = 0
        
        # 从后向前计算折扣奖励
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted[t] = running_add
        
        # 归一化奖励
        if len(discounted) > 0 and np.std(discounted) > 0:
            discounted = (discounted - np.mean(discounted)) / np.std(discounted)
        
        return discounted
    
    def train(self) -> float:
        """训练策略网络"""
        if len(self.states) == 0:
            return 0.0
        
        # 计算折扣奖励
        discounted_rewards = self.discount_rewards(self.rewards)
        
        # 准备训练数据
        states = np.array(self.states)
        actions = np.array(self.actions)
        
        # 创建动作标签（one-hot编码）
        action_labels = np.zeros((len(actions), self.action_dim))
        for i, action in enumerate(actions):
            action_labels[i, action] = discounted_rewards[i]
        
        # 训练模型
        loss = self.policy_network.train_on_batch(states, action_labels)
        
        # 清空轨迹数据
        self.states = []
        self.actions = []
        self.rewards = []
        
        return loss
    
    def remember_reward(self, reward: float) -> None:
        """记录奖励"""
        self.rewards.append(reward)
    
    def load(self, name: str) -> None:
        """加载模型权重"""
        self.policy_network.load_weights(name)
    
    def save(self, name: str) -> None:
        """保存模型权重"""
        self.policy_network.save_weights(name)
    
    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """获取动作概率分布"""
        return self.policy_network.predict(state.reshape(1, -1), verbose=0)[0]

# 示例使用
if __name__ == "__main__":
    from ..game.poker_game import Player, PokerGame
    
    # 创建玩家
    players = [
        Player(f"Player {i}", 1000, i > 0) for i in range(6)
    ]
    
    # 创建游戏
    game = PokerGame(5, 10, players)
    game.start_new_hand()
    
    # 创建状态表示器
    state_rep = StateRepresentation()
    
    # 获取状态维度
    state_dim = state_rep.total_state_dim
    
    # 创建DQN代理
    dqn_agent = DQNAgent(state_dim=state_dim)
    
    # 创建策略梯度代理
    pg_agent = PolicyGradientAgent(state_dim=state_dim)
    
    # 获取当前状态
    current_state = state_rep.get_state(game, players[0])
    
    # DQN代理选择动作
    dqn_action = dqn_agent.act(current_state, game, players[0])
    print(f"DQN代理选择的动作: {dqn_action}")
    
    # 策略梯度代理选择动作
    pg_action = pg_agent.act(current_state, game, players[0])
    print(f"策略梯度代理选择的动作: {pg_action}")