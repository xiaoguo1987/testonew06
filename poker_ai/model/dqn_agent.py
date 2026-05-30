# 德州扑克DQN代理模型 (最终完整版)

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adamax
from collections import deque
import random
from typing import List, Tuple, Dict, Optional
from .state_representation import StateRepresentation
from ..game.poker_game import Player, PokerGame
import logging

logger = logging.getLogger("DQNAgent")

# ==========================================
# 1. DQN Agent (已修复优化器和保存逻辑)
# ==========================================
class DQNAgent:
    """德州扑克DQN智能代理"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int = 5,
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
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
        
        # 显式初始化优化器
        self.optimizer = self._get_optimizer()
        
        # 创建主网络和目标网络
        # 注意：mixed precision 下，同一个 optimizer 不能被重复 wrap。
        # target_model 仅用于推理目标 Q，不参与训练，因此不需要 compile。
        self.model = self._build_model(compile_model=True)
        self.target_model = self._build_model(compile_model=False)
        self.update_target_model() 
        
        self.state_rep = StateRepresentation()
        self.training_steps = 0

        # 预编译推理函数
        self.predict_fn = tf.function(
            self._predict_internal,
            input_signature=[tf.TensorSpec(shape=(None, self.state_dim), dtype=tf.float32)]
        )
        
        # 预编译训练函数
        self.train_step = tf.function(
            self._train_step_internal,
            input_signature=[
                tf.TensorSpec(shape=(None, self.state_dim), dtype=tf.float32), 
                tf.TensorSpec(shape=(None, self.action_dim), dtype=tf.float32)
            ]
        )

    def _get_optimizer(self):
        """获取优化器实例"""
        name = self.optimizer_type.lower()
        if name == 'adam': return Adam(learning_rate=self.learning_rate)
        if name == 'adamax': return Adamax(learning_rate=self.learning_rate)
        if name == 'rmsprop': return RMSprop(learning_rate=self.learning_rate)
        if name == 'sgd': return SGD(learning_rate=self.learning_rate)
        return Adam(learning_rate=self.learning_rate)

    def _predict_internal(self, state_tensor):
        return self.model(state_tensor, training=False)

    def _train_step_internal(self, states, target_q):
        with tf.GradientTape() as tape:
            predictions = self.model(states, training=True)
            loss = tf.keras.losses.MeanSquaredError()(target_q, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def _build_model(self, compile_model: bool = True) -> Model:
        model = Sequential([
            Input(shape=(self.state_dim,)),
            Dense(512, activation='relu'),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(self.action_dim, activation='linear')
        ])
        if compile_model:
            model.compile(optimizer=self.optimizer, loss='mse')
        return model
    
    def update_target_model(self) -> None:
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))
    
    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)
        self.update_target_model()

    def act(self, state: np.ndarray, game: Optional[PokerGame] = None, player: Optional[Player] = None) -> Tuple[str, Optional[int]]:
        if np.random.rand() <= self.epsilon:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
            q_values_tensor = self.predict_fn(state_tensor)
            q_values = q_values_tensor.numpy()[0]
            action_idx = np.argmax(q_values)
        
        if game is None or player is None:
            return action_idx, None
        else:
            action_vector = np.zeros(self.action_dim)
            action_vector[action_idx] = 1
            return self.state_rep.decode_action(action_vector, game, player, 0.0)
            
    def replay(self) -> None:
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([exp[0] for exp in minibatch])
        actions = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch])
        
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)
        
        target_q = self.model(states_tensor, training=False).numpy()
        next_q = self.target_model(next_states_tensor, training=False).numpy()
        
        for i in range(self.batch_size):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                target_q[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
        
        target_q_tensor = tf.convert_to_tensor(target_q, dtype=tf.float32)
        self.train_step(states_tensor, target_q_tensor)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.update_target_model()

    def load(self, name: str) -> None:
        """加载模型权重 (安全版)"""
        h5_path = name
        if name.endswith('.pkl'):
            h5_path = name.replace('.pkl', '.weights.h5')
        elif not name.endswith('.weights.h5'):
            h5_path = name + '.weights.h5'
            
        if os.path.exists(h5_path):
            try:
                self.model.load_weights(h5_path)
                self.update_target_model()
                print(f"✅ 成功加载权重: {h5_path}")
            except Exception as e:
                print(f"❌ 加载权重失败: {e}")
        else:
            print(f"⚠️ 权重文件不存在: {h5_path}")
    
    def save(self, name: str) -> None:
        """保存模型权重"""
        if name.endswith('.pkl'):
            name = name.replace('.pkl', '.weights.h5')
        elif not name.endswith('.weights.h5'):
            name = name + '.weights.h5'
            
        try:
            self.model.save_weights(name)
        except Exception as e:
            print(f"❌ 保存权重失败: {e}")
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        return self.model(state_tensor, training=False).numpy()[0]


# ==========================================
# 2. Policy Gradient Agent (必须包含！)
# ==========================================
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
        
        self.policy_network = self._build_policy_network()
        self.state_rep = StateRepresentation()
        
        self.states = []
        self.actions = []
        self.rewards = []

        # 同样使用 tf.function 预编译推理，保持一致的高性能
        self.predict_fn = tf.function(
            self._predict_internal,
            input_signature=[tf.TensorSpec(shape=(None, self.state_dim), dtype=tf.float32)]
        )

    def _predict_internal(self, state_tensor):
        return self.policy_network(state_tensor, training=False)
    
    def _build_policy_network(self) -> Model:
        model = Sequential([
            Input(shape=(self.state_dim,)),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(self.action_dim, activation='softmax')
        ])
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy'
        )
        return model
    
    def act(self, state: np.ndarray, game: Optional[PokerGame] = None, player: Optional[Player] = None) -> Tuple[str, Optional[int]]:
        # 使用预编译函数加速
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        action_probs = self.predict_fn(state_tensor).numpy()[0]
        
        action_idx = np.random.choice(self.action_dim, p=action_probs)
        
        self.states.append(state)
        self.actions.append(action_idx)
        
        if game is None or player is None:
            return action_idx, None
        else:
            action_vector = np.zeros(self.action_dim)
            action_vector[action_idx] = 1
            return self.state_rep.decode_action(action_vector, game, player, 0.0)

    def discount_rewards(self, rewards: List[float]) -> np.ndarray:
        discounted = np.zeros_like(rewards, dtype=float)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted[t] = running_add
        
        if len(discounted) > 0 and np.std(discounted) > 0:
            discounted = (discounted - np.mean(discounted)) / np.std(discounted)
        return discounted
    
    def train(self) -> float:
        if len(self.states) == 0:
            return 0.0
        
        discounted_rewards = self.discount_rewards(self.rewards)
        states = np.array(self.states)
        actions = np.array(self.actions)
        
        action_labels = np.zeros((len(actions), self.action_dim))
        for i, action in enumerate(actions):
            action_labels[i, action] = discounted_rewards[i]
        
        loss = self.policy_network.train_on_batch(states, action_labels)
        
        self.states = []
        self.actions = []
        self.rewards = []
        return loss
    
    def remember_reward(self, reward: float) -> None:
        self.rewards.append(reward)
    
    def load(self, name: str) -> None:
        if not name.endswith('.weights.h5'):
             name += '.weights.h5'
        if os.path.exists(name):
             self.policy_network.load_weights(name)
    
    def save(self, name: str) -> None:
        if not name.endswith('.weights.h5'):
             name += '.weights.h5'
        self.policy_network.save_weights(name)
    
    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        return self.policy_network(state_tensor, training=False).numpy()[0]
