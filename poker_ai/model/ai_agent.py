# 德州扑克AI代理接口

import numpy as np
from typing import Tuple, Optional, Dict
from .state_representation import StateRepresentation
from .dqn_agent import DQNAgent, PolicyGradientAgent
from ..game.poker_game import Player, PokerGame

class PokerAIAgent:
    """德州扑克AI代理统一接口"""
    
    def __init__(self, agent_type: str = 'dqn', state_dim: Optional[int] = None, target_update_freq: int = 10):
        """
        初始化AI代理
        agent_type: 'dqn' 或 'policy_gradient'
        state_dim: 状态维度，如果为None则自动计算
        target_update_freq: DQN目标网络更新频率（仅DQN代理有效）
        """
        self.state_rep = StateRepresentation()
        
        # 如果没有指定状态维度，使用默认值
        if state_dim is None:
            state_dim = self.state_rep.total_state_dim
        
        # 根据类型创建相应的代理
        if agent_type == 'dqn':
            self.agent = DQNAgent(state_dim=state_dim, target_update_freq=target_update_freq)
        elif agent_type == 'policy_gradient':
            self.agent = PolicyGradientAgent(state_dim=state_dim)
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
        
        # 添加游戏状态信息
        decision_info['current_bet'] = game.current_bet
        decision_info['player_bet'] = player.bet
        decision_info['pot_size'] = game.pot
        decision_info['round'] = game.round
        
        return decision_info
    
    def load_model(self, filepath: str) -> None:
        """加载模型权重"""
        self.agent.load(filepath)
    
    def save_model(self, filepath: str) -> None:
        """保存模型权重"""
        self.agent.save(filepath)
    
    def update_target_model(self) -> None:
        """更新目标模型（DQN特定）"""
        if self.agent_type == 'dqn':
            self.agent.update_target_model()
    
    def get_epsilon(self) -> Optional[float]:
        """获取探索率（DQN特定）"""
        if self.agent_type == 'dqn':
            return self.agent.epsilon
        return None
    
    def set_epsilon(self, value: float) -> None:
        """设置探索率（DQN特定）"""
        if self.agent_type == 'dqn' and hasattr(self.agent, 'epsilon'):
            self.agent.epsilon = max(0.1, min(1.0, value))

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
    
    # 创建集成代理
    ensemble_agent = EnsembleAgent()
    
    # 测试动作选择
    ai_action = ai_agent.select_action(game, players[1])
    print(f"单个AI代理选择的动作: {ai_action}")
    
    ensemble_action = ensemble_agent.select_action(game, players[2])
    print(f"集成代理选择的动作: {ensemble_action}")
    
    # 测试决策信息
    decision_info = ai_agent.get_decision_info(game, players[1])
    print(f"\n决策信息示例: {decision_info.keys()}")
    
    if 'q_values' in decision_info:
        print(f"Q值: {decision_info['q_values']}")
    elif 'action_probabilities' in decision_info:
        print(f"动作概率: {decision_info['action_probabilities']}")