# 德州扑克奖励计算模块

import numpy as np
from typing import Dict, List, Tuple, Optional
from ..game.poker_game import PokerGame, Player
from ..game.cards import evaluate_hand, compare_hands

class RewardCalculator:
    """德州扑克奖励计算器"""
    
    def __init__(self):
        # 定义不同奖励类型的权重
        self.reward_weights = {
            'winning_hand': 1.0,          # 赢牌奖励
            'losing_hand': -0.8,          # 输牌惩罚
            'pot_odds': 0.2,              # 底池赔率奖励
            'hand_strength': 0.3,         # 手牌强度奖励
            'bluff_success': 0.4,         # 成功诈唬奖励
            'fold_when_weak': 0.3,        # 及时弃牌奖励
            'aggressive_play': 0.1,       # 主动进攻奖励
            'position_advantage': 0.15,   # 位置优势奖励
            'bankroll_change': 0.5        # 筹码变化奖励
        }
        
        # 定义不同游戏阶段的奖励调整因子
        self.stage_factors = {
            1: 1.0,  # 翻牌前
            2: 1.2,  # 翻牌
            3: 1.5,  # 转牌
            4: 2.0   # 河牌
        }
    
    def calculate_hand_strength(self, player: Player, game: PokerGame) -> float:
        """
        估算手牌强度（0-1之间）
        """
        if not player.is_active:
            return 0.0
        
        # 如果有公共牌，使用蒙特卡洛模拟估算胜率
        if game.board:
            # 这里简化实现，实际应该使用蒙特卡洛模拟
            # 或更复杂的手牌强度算法
            return self._estimate_win_probability(player, game)
        else:
            # 翻牌前的手牌强度评估
            return self._preflop_hand_strength(player.cards)
    
    def _preflop_hand_strength(self, cards: List) -> float:
        """
        翻牌前的手牌强度评估
        """
        if len(cards) != 2:
            return 0.0
        
        # 获取两张牌的点数和是否同色
        rank1, rank2 = cards[0].rank, cards[1].rank
        is_suited = cards[0].suit == cards[1].suit
        
        # 定义一些常见起手牌的相对强度
        # 这里是一个简化版本，实际应该使用更精确的评估
        hand_strength_map = {
            ('A', 'A'): 0.95,
            ('K', 'K'): 0.85,
            ('Q', 'Q'): 0.75,
            ('J', 'J'): 0.65,
            ('T', 'T'): 0.55,
            ('9', '9'): 0.45,
            ('A', 'K'): 0.8 if is_suited else 0.65,
            ('A', 'Q'): 0.75 if is_suited else 0.6,
            ('A', 'J'): 0.7 if is_suited else 0.55,
            ('A', 'T'): 0.65 if is_suited else 0.5,
            ('K', 'Q'): 0.65 if is_suited else 0.5,
            ('K', 'J'): 0.6 if is_suited else 0.45,
            ('Q', 'J'): 0.55 if is_suited else 0.4,
            # 更多组合可以继续添加
        }
        
        # 考虑牌的顺序无关性
        key = tuple(sorted([rank1, rank2], reverse=True))
        base_strength = hand_strength_map.get(key, 0.3)
        
        # 如果是同色但不在映射表中，增加一点强度
        if is_suited and key not in hand_strength_map:
            base_strength += 0.1
        
        # 如果是连续牌，增加一点强度
        if abs(self._rank_to_value(rank1) - self._rank_to_value(rank2)) == 1:
            base_strength += 0.1
        
        return min(1.0, max(0.0, base_strength))
    
    def _rank_to_value(self, rank: str) -> int:
        """将牌点转换为数值"""
        rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                      '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        return rank_values.get(rank, 0)
    
    def _estimate_win_probability(self, player: Player, game: PokerGame) -> float:
        """
        使用简化的方法估算胜率
        在实际实现中，这里应该使用蒙特卡洛模拟
        """
        # 简化版本 - 根据已有的公共牌和手牌评估强度
        # 实际实现中，应该模拟剩余的公共牌和对手的可能手牌
        
        # 计算当前的手牌等级
        if len(game.board) >= 3:
            # 使用当前可见的牌进行评估
            best_hand = player.cards + game.board
            # 这里简化处理，返回一个基于手牌等级的强度值
            # 实际中应该使用更复杂的算法
            return 0.5 + (len(game.board) / 5) * 0.3
        
        return 0.5  # 默认值
    
    def calculate_pot_odds(self, player: Player, game: PokerGame, action_amount: Optional[int] = None) -> float:
        """
        计算底池赔率奖励
        好的底池赔率应该得到正奖励
        """
        if action_amount is None or action_amount <= 0:
            return 0.0
        
        # 计算底池赔率 = 底池大小 / 所需下注
        pot_size = game.pot
        odds = pot_size / action_amount if action_amount > 0 else 0
        
        # 转换为奖励值，好的赔率应该为正
        # 当赔率 > 1.5时认为是好赔率
        if odds > 3.0:
            return 1.0
        elif odds > 1.5:
            return 0.5
        elif odds > 1.0:
            return 0.0
        else:
            return -0.5
    
    def calculate_position_advantage(self, player: Player, game: PokerGame) -> float:
        """
        计算位置优势奖励
        后行动有位置优势
        """
        # 计算玩家相对于庄家的位置
        # 0: 庄家, 1: 小盲, 2: 大盲, 3-5: 其他位置
        position = player.position
        dealer_pos = game.dealer_pos
        
        # 计算相对位置
        relative_pos = (position - dealer_pos) % 6
        
        # 位置奖励：位置越靠后奖励越高
        position_rewards = {
            0: 0.8,  # 庄家
            1: -0.5, # 小盲
            2: -0.2, # 大盲
            3: 0.1,  # 枪口位置+1
            4: 0.4,  # 中间位置
            5: 0.6   # 枪口位置
        }
        
        return position_rewards.get(relative_pos, 0.0)
    
    def is_bluffing(self, player: Player, game: PokerGame, action: str) -> bool:
        """
        判断玩家是否在诈唬
        简化版本：手牌强度低但仍在进攻
        """
        hand_strength = self.calculate_hand_strength(player, game)
        
        # 如果手牌强度低但选择加注，则可能是诈唬
        if hand_strength < 0.3 and action == 'raise':
            return True
        
        return False
    
    def calculate_reward(self, player: Player, game: PokerGame, action: str, 
                         action_amount: Optional[int] = None, 
                         is_winner: bool = False, 
                         profit_loss: float = 0.0) -> float:
        """
        计算综合奖励
        """
        total_reward = 0.0
        
        # 1. 筹码变化奖励
        bankroll_reward = profit_loss / (game.big_blind * 10)  # 归一化到10个大盲注
        total_reward += self.reward_weights['bankroll_change'] * bankroll_reward
        
        # 2. 赢牌/输牌奖励
        if is_winner:
            total_reward += self.reward_weights['winning_hand']
        else:
            # 只有在玩家参与到摊牌时才惩罚输牌
            active_players = [p for p in game.players if p.is_active]
            if player.is_active and len(active_players) > 1:
                total_reward += self.reward_weights['losing_hand']
        
        # 3. 手牌强度奖励
        hand_strength = self.calculate_hand_strength(player, game)
        total_reward += self.reward_weights['hand_strength'] * hand_strength
        
        # 4. 底池赔率奖励
        if action in ['call', 'raise'] and action_amount is not None:
            pot_odds_reward = self.calculate_pot_odds(player, game, action_amount)
            total_reward += self.reward_weights['pot_odds'] * pot_odds_reward
        
        # 5. 位置优势奖励
        position_reward = self.calculate_position_advantage(player, game)
        total_reward += self.reward_weights['position_advantage'] * position_reward
        
        # 6. 主动进攻奖励
        if action == 'raise':
            total_reward += self.reward_weights['aggressive_play']
        
        # 7. 及时弃牌奖励
        if action == 'fold' and hand_strength < 0.2:
            total_reward += self.reward_weights['fold_when_weak']
        
        # 8. 成功诈唬奖励
        if self.is_bluffing(player, game, action) and is_winner:
            total_reward += self.reward_weights['bluff_success']
        
        # 根据游戏阶段调整奖励
        stage_factor = self.stage_factors.get(game.round, 1.0)
        total_reward *= stage_factor
        
        # 限制奖励范围
        return max(-2.0, min(2.0, total_reward))
    
    def calculate_episode_rewards(self, player: Player, game: PokerGame, 
                                action_history: List[Tuple[str, Optional[int]]],
                                is_winner: bool, profit_loss: float) -> List[float]:
        """
        计算一整局游戏中每个动作的奖励
        这对于回合制的强化学习方法很有用
        """
        rewards = []
        
        # 为每个动作计算即时奖励
        for action, amount in action_history:
            immediate_reward = self.calculate_reward(
                player, game, action, amount, is_winner=False, profit_loss=0.0
            )
            rewards.append(immediate_reward)
        
        # 最后加上最终奖励（赢牌/输牌和筹码变化）
        if rewards:
            final_reward = self.calculate_reward(
                player, game, action_history[-1][0], action_history[-1][1], 
                is_winner, profit_loss
            )
            rewards[-1] += final_reward
        
        return rewards

# 动态奖励调整器
class AdaptiveRewardCalculator:
    """
    自适应奖励计算器，可以根据AI的表现动态调整奖励权重
    """
    
    def __init__(self, base_calculator: Optional[RewardCalculator] = None):
        self.base_calculator = base_calculator or RewardCalculator()
        self.performance_history = []  # 存储最近的表现
        self.adjustment_factor = 0.05  # 调整因子
        self.history_size = 50  # 历史记录大小
    
    def record_performance(self, profit_loss: float, hands_played: int) -> None:
        """
        记录AI的表现
        """
        # 计算每手牌的平均利润
        profit_per_hand = profit_loss / hands_played if hands_played > 0 else 0
        self.performance_history.append(profit_per_hand)
        
        # 保持历史记录大小
        if len(self.performance_history) > self.history_size:
            self.performance_history.pop(0)
    
    def adjust_rewards(self) -> None:
        """
        根据最近表现调整奖励权重
        """
        if len(self.performance_history) < 10:
            return  # 数据不足，不调整
        
        # 计算最近表现趋势
        recent_avg = np.mean(self.performance_history[-5:])
        overall_avg = np.mean(self.performance_history)
        
        # 如果最近表现下降，调整奖励权重
        if recent_avg < overall_avg * 0.8:
            # 增加一些探索性的奖励权重
            self.base_calculator.reward_weights['aggressive_play'] += self.adjustment_factor
            self.base_calculator.reward_weights['bluff_success'] += self.adjustment_factor * 0.5
            
            # 减少对保守策略的奖励
            if self.base_calculator.reward_weights['fold_when_weak'] > 0.1:
                self.base_calculator.reward_weights['fold_when_weak'] -= self.adjustment_factor * 0.5
        
        # 限制权重范围
        for key in self.base_calculator.reward_weights:
            self.base_calculator.reward_weights[key] = max(0.0, min(2.0, self.base_calculator.reward_weights[key]))
    
    def calculate_reward(self, *args, **kwargs) -> float:
        """
        计算奖励（与基础计算器相同）
        """
        return self.base_calculator.calculate_reward(*args, **kwargs)

# 示例使用
if __name__ == "__main__":
    from ..game.poker_game import Player, PokerGame
    from ..game.cards import Card
    
    # 创建玩家
    players = [Player(f"Player {i}", 1000, i > 0) for i in range(6)]
    
    # 创建游戏
    game = PokerGame(5, 10, players)
    game.start_new_hand()
    
    # 创建奖励计算器
    reward_calc = RewardCalculator()
    
    # 测试手牌强度计算
    player = players[0]
    print(f"玩家 {player.name} 的手牌: {player.cards}")
    hand_strength = reward_calc.calculate_hand_strength(player, game)
    print(f"手牌强度: {hand_strength:.2f}")
    
    # 测试奖励计算
    reward = reward_calc.calculate_reward(player, game, 'call', 10, is_winner=False, profit_loss=-10)
    print(f"奖励值: {reward:.2f}")
    
    # 测试自适应奖励计算器
    adaptive_calc = AdaptiveRewardCalculator()
    adaptive_calc.record_performance(-100, 20)  # 记录负利润
    adaptive_calc.adjust_rewards()
    print("调整后的奖励权重:")
    print(adaptive_calc.base_calculator.reward_weights)