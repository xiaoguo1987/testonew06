# 德州扑克AI代理接口

import numpy as np
import logging
from typing import Tuple, Optional, Dict
from .state_representation import StateRepresentation
from .dqn_agent import DQNAgent, PolicyGradientAgent
from .ppo_model import PPOModel  # 添加PPO模型导入
from ..game.poker_game import Player, PokerGame

# 设置日志
logger = logging.getLogger('PokerAIAgent')

class PokerAIAgent:
    """德州扑克AI代理统一接口"""
    
    def __init__(self, agent_type: str = 'dqn', state_dim: Optional[int] = None, target_update_freq: int = 10, 
                 learning_rate: float = 0.001, optimizer: str = 'adam'):
        """
        初始化AI代理
        agent_type: 'dqn'、'policy_gradient' 或 'ppo'
        state_dim: 状态维度，如果为None则自动计算
        target_update_freq: DQN目标网络更新频率（仅DQN代理有效）
        learning_rate: 学习率
        optimizer: 优化器类型
        """
        self.state_rep = StateRepresentation()
        
        # 如果没有指定状态维度，使用默认值
        if state_dim is None:
            state_dim = self.state_rep.total_state_dim
        
        # 根据类型创建相应的代理
        if agent_type == 'dqn':
            self.agent = DQNAgent(state_dim=state_dim, target_update_freq=target_update_freq, 
                                 learning_rate=learning_rate, optimizer=optimizer)
        elif agent_type == 'policy_gradient':
            self.agent = PolicyGradientAgent(state_dim=state_dim)
        elif agent_type == 'ppo':
            # 创建PPO模型
            self.agent = PPOModel(input_dim=state_dim, lr=learning_rate)
        else:
            raise ValueError(f"不支持的代理类型: {agent_type}")
        
        self.agent_type = agent_type
        self.name = f"AI_Agent_{agent_type}"
    
    def select_action(self, game: PokerGame, player: Player) -> Tuple[str, Optional[int]]:
        """
        根据当前游戏状态选择动作
        返回: (动作类型, 动作金额)
        """
        # 获取当前状态
        state = self.state_rep.get_state(game, player)
        
        # 选择动作
        return self.agent.act(state, game, player)
    
    def learn(self, reward: float, state: Optional[np.ndarray] = None, 
              action: Optional[int] = None, next_state: Optional[np.ndarray] = None, 
              done: Optional[bool] = False) -> None:
        """
        从经验中学习
        """
        if self.agent_type == 'dqn' and all(v is not None for v in [state, action, next_state, done]):
            # DQN代理需要完整的经验元组
            self.agent.remember(state, action, reward, next_state, done)
            self.agent.replay()
        elif self.agent_type == 'policy_gradient':
            # 策略梯度代理只需要记住奖励
            self.agent.remember_reward(reward)
            # 回合结束时训练
            if done:
                self.agent.train()
        elif self.agent_type == 'ppo':
            # PPO代理需要完整的经验元组
            if all(v is not None for v in [state, action, next_state, done]):
                # 对于PPO，累积经验
                if not hasattr(self, 'ppo_buffer'):
                    # 初始化PPO经验缓冲区
                    self.ppo_buffer = {
                        'states': [],
                        'actions': [],
                        'rewards': [],
                        'dones': [],
                        'old_probs': []
                    }
                
                # 获取旧动作概率
                old_probs, _ = self.agent.predict(state)
                old_prob = old_probs[action]
                
                # 添加到缓冲区
                self.ppo_buffer['states'].append(state)
                self.ppo_buffer['actions'].append(action)
                self.ppo_buffer['rewards'].append(reward)
                self.ppo_buffer['dones'].append(done)
                self.ppo_buffer['old_probs'].append(old_prob)
                
                # 如果是回合结束，训练PPO模型
                if done and len(self.ppo_buffer['states']) > 0:
                    # 将缓冲区转换为numpy数组
                    states = np.array(self.ppo_buffer['states'])
                    actions = np.array(self.ppo_buffer['actions'])
                    rewards = np.array(self.ppo_buffer['rewards'])
                    dones = np.array(self.ppo_buffer['dones'])
                    old_probs = np.array(self.ppo_buffer['old_probs'])
                    
                    # 训练PPO模型
                    self.agent.train(states, actions, rewards, dones, old_probs)
                    
                    # 清空缓冲区
                    self.ppo_buffer = {
                        'states': [],
                        'actions': [],
                        'rewards': [],
                        'dones': [],
                        'old_probs': []
                    }
    
    def get_decision_info(self, game: PokerGame, player: Player) -> Dict:
        """
        获取决策信息（用于提供建议）
        返回: 包含决策依据的字典
        """
        state = self.state_rep.get_state(game, player)
        decision_info = {}
        
        if self.agent_type == 'dqn':
            # 获取Q值
            q_values = self.agent.get_q_values(state)
            decision_info['q_values'] = q_values.tolist()
            decision_info['best_action_index'] = int(np.argmax(q_values))
        elif self.agent_type == 'policy_gradient':
            # 获取动作概率
            action_probs = self.agent.get_action_probs(state)
            decision_info['action_probabilities'] = action_probs.tolist()
            decision_info['most_likely_action'] = int(np.argmax(action_probs))
        elif self.agent_type == 'ppo':
            # 获取动作概率和状态价值
            action_probs, value = self.agent.predict(state)
            decision_info['action_probabilities'] = action_probs.tolist()
            decision_info['state_value'] = float(value)
            decision_info['most_likely_action'] = int(np.argmax(action_probs))
        
        # 添加游戏状态信息
        decision_info['current_bet'] = game.current_bet
        decision_info['player_bet'] = player.bet
        decision_info['pot_size'] = game.pot
        decision_info['round'] = game.round
        
        return decision_info
    
    def load_model(self, filepath: str) -> bool:
        """加载模型权重"""
        try:
            # 必须调用 self.agent.load，而不是 self.load_model
            self.agent.load(filepath) 
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def save_model(self, filepath: str) -> None:
        """保存模型权重"""
        self.agent.save(filepath)
    
    # === 新增：兼容 CheckpointManager 的别名方法 ===
    def save(self, filepath: str) -> None:
        """兼容性接口：调用 save_model"""
        self.save_model(filepath)

    def load(self, filepath: str) -> None:
        """兼容性接口：调用 load_model"""
        self.load_model(filepath)
    # ============================================

    def get_weights(self):
        """获取代理权重"""
        if hasattr(self.agent, 'get_weights'):
            return self.agent.get_weights()
        return None

    def set_weights(self, weights):
        """设置代理权重"""
        if hasattr(self.agent, 'set_weights'):
            self.agent.set_weights(weights)

    def update_target_model(self) -> None:
        """更新目标模型（DQN特定）"""
        if self.agent_type == 'dqn':
            self.agent.update_target_model()
    
    def get_epsilon(self) -> Optional[float]:
        """获取探索率（DQN特定）"""
        if self.agent_type == 'dqn':
            return self.agent.epsilon
        return None
    
    def set_epsilon(self, value: float, min_value: Optional[float] = None, decay_rate: Optional[float] = None) -> None:
        """设置探索参数（DQN特定）
        
        Args:
            value: 当前探索率值
            min_value: 最小探索率值（可选）
            decay_rate: 衰减率值（可选）
        """
        if self.agent_type == 'dqn' and hasattr(self.agent, 'epsilon'):
            # 设置当前探索率，确保在合理范围内
            self.agent.epsilon = max(0.01, min(1.0, value))
            
            # 设置最小探索率（如果提供）
            if min_value is not None and hasattr(self.agent, 'epsilon_min'):
                self.agent.epsilon_min = max(0.01, min(1.0, min_value))
            
            # 设置衰减率（如果提供）
            if decay_rate is not None and hasattr(self.agent, 'epsilon_decay'):
                self.agent.epsilon_decay = max(0.00001, min(0.1, decay_rate))
            
            # 日志记录
            logger.info(f"更新DQN探索参数 - epsilon: {self.agent.epsilon}, " 
                      f"epsilon_min: {getattr(self.agent, 'epsilon_min', 'N/A')}, "
                      f"epsilon_decay: {getattr(self.agent, 'epsilon_decay', 'N/A')}")

# 创建一个组合多个AI代理的集成模型
class EnsembleAgent:
    """集成多个AI代理的决策模型"""
    
    def __init__(self, agent_types: list = ['dqn', 'policy_gradient']):
        """
        初始化集成代理
        agent_types: 要集成的代理类型列表
        """
        state_rep = StateRepresentation()
        state_dim = state_rep.total_state_dim
        
        self.agents = []
        for agent_type in agent_types:
            self.agents.append(PokerAIAgent(agent_type=agent_type, state_dim=state_dim))
        
        self.state_rep = state_rep
    
    def select_action(self, game: PokerGame, player: Player) -> Tuple[str, Optional[int]]:
        """
        通过投票机制选择最终动作
        """
        # 收集所有代理的动作建议
        actions = []
        for agent in self.agents:
            action = agent.select_action(game, player)
            actions.append(action[0])  # 只考虑动作类型
        
        # 统计动作类型的投票数
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # 选择得票最多的动作类型
        best_action_type = max(action_counts.keys(), key=lambda x: action_counts[x])
        
        # 如果选中的是raise，使用第一个代理的金额建议
        if best_action_type == 'raise':
            for agent in self.agents:
                action = agent.select_action(game, player)
                if action[0] == 'raise' and action[1] is not None:
                    return (best_action_type, action[1])
        elif best_action_type == 'call':
            # 跟注金额固定
            call_amount = max(0, game.current_bet - player.bet)
            return (best_action_type, call_amount)
        
        return (best_action_type, None)
    
    def learn(self, reward: float, done: bool = False) -> None:
        """
        更新所有代理
        """
        for agent in self.agents:
            # 对于集成模型，简化学习过程
            agent.learn(reward, done=done)
    
    def get_decision_info(self, game: PokerGame, player: Player) -> Dict:
        """
        获取集成决策信息
        """
        info = {
            'ensemble_decisions': [],
            'consensus': {}
        }
        
        # 收集每个代理的决策信息
        for i, agent in enumerate(self.agents):
            agent_info = agent.get_decision_info(game, player)
            agent_info['agent_type'] = agent.agent_type
            info['ensemble_decisions'].append(agent_info)
        
        # 计算动作投票
        actions = []
        for agent in self.agents:
            action = agent.select_action(game, player)
            actions.append(action[0])
        
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        info['consensus']['action_votes'] = action_counts
        
        return info
    
    def save_models(self, base_path: str) -> None:
        """
        保存所有代理的模型
        """
        for i, agent in enumerate(self.agents):
            filepath = f"{base_path}_{agent.agent_type}_{i}.h5"
            agent.save_model(filepath)
    
    def load_models(self, base_path: str) -> None:
        """
        加载所有代理的模型
        """
        for i, agent in enumerate(self.agents):
            filepath = f"{base_path}_{agent.agent_type}_{i}.h5"
            try:
                agent.load_model(filepath)
            except Exception as e:
                print(f"无法加载模型 {filepath}: {e}")

# 示例使用
if __name__ == "__main__":
    from ..game.poker_game import Player, PokerGame
    
    # 创建玩家
    players = [
        Player("Human", 1000, False),
        Player("AI_1", 1000, True),
        Player("AI_2", 1000, True),
        Player("AI_3", 1000, True),
        Player("AI_4", 1000, True),
        Player("AI_5", 1000, True)
    ]
    
    # 创建游戏
    game = PokerGame(5, 10, players)
    game.start_new_hand()
    
    # 创建单个AI代理
    ai_agent = PokerAIAgent(agent_type='dqn')
    
    # 创建PPO代理
    ppo_agent = PokerAIAgent(agent_type='ppo')
    
    # 创建集成代理
    ensemble_agent = EnsembleAgent()
    
    # 测试动作选择
    ai_action = ai_agent.select_action(game, players[1])
    print(f"单个AI代理选择的动作: {ai_action}")
    
    ppo_action = ppo_agent.select_action(game, players[2])
    print(f"PPO代理选择的动作: {ppo_action}")
    
    ensemble_action = ensemble_agent.select_action(game, players[3])
    print(f"集成代理选择的动作: {ensemble_action}")
    
    # 测试决策信息
    decision_info = ai_agent.get_decision_info(game, players[1])
    print(f"\n决策信息示例: {decision_info.keys()}")
    
    if 'q_values' in decision_info:
        print(f"Q值: {decision_info['q_values']}")
    elif 'action_probabilities' in decision_info:
        print(f"动作概率: {decision_info['action_probabilities']}")