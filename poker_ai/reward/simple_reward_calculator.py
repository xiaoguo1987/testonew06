import numpy as np
from typing import Optional
from ..game.poker_game import PokerGame, Player

class SimpleRewardCalculator:
    """
    极简奖励计算器 - 专注于最终结果 (End-to-End Learning)
    同时保留辅助方法用于日志统计
    """
    def __init__(self):
        pass
        
    def calculate_reward(self, player: Player, game: PokerGame, action: str, 
                         amount: int, is_winner: bool, profit_loss: float, done: bool) -> float:
        """
        只在游戏结束时给予基于盈亏的奖励。
        过程中的奖励为 0。
        """
        if not done:
            strength = self.calculate_hand_strength(player, game)
            # 🔥 浪打惩罚：牌烂还要玩
            if action in ['call', 'raise'] and strength < 0.4:
                return -0.2  # 惩罚浪打
            # 注意：这里需要调用 calculate_hand_strength，确保该方法在类里可用
            if action == 'fold':
                strength = self.calculate_hand_strength(player, game)
                if strength > 0.7:  # 强牌弃牌 -> 重罚
                    return -1.2
                elif strength > 0.5: # 中等牌弃牌 -> 轻罚
                    return -0.5
                else:
                    return -0.05 # 烂牌弃牌 -> 轻微扣分 (正常的盲注损耗)
            return 0.0
        reward = 0.0    
        # 归一化：以大盲注为单位
        # 赢 10BB 就是 +10 分，输 10BB 就是 -10 分
        if game.big_blind > 0:
            # 赢钱给奖励
            if profit_loss > 0:
                reward += (profit_loss / game.big_blind) * 1.0 
            # 输钱加倍惩罚！让它知道痛！
            else:
                reward += (profit_loss / game.big_blind) * 1.5
        
        if is_winner:
            # 以前是固定 +5.0，现在改为 +2.0 加上底池比例奖励
            # 这样鼓励它赢大底池，而不是只赢盲注
            pot_bonus = (game.pot / game.big_blind) * 0.1
            reward += (3.5 + pot_bonus)
            # 4. 输牌安慰奖 (防止它因为怕输而不敢玩)
        # 裁剪奖励防止梯度爆炸 (-5 到 +5 之间)
        return max(-10.0, min(10.0, reward))

    # --- 以下是为了兼容 Trainer 日志统计而补全的辅助方法 ---

    def calculate_hand_strength(self, player: Player, game: PokerGame) -> float:
        """
        估算手牌强度（用于统计日志，非奖励计算核心）
        简单起见，我们可以直接调用 state_representation 中的逻辑，或者写一个简化版
        """
        # 如果你想复用复杂的逻辑，可以从原来的 RewardCalculator 复制
        # 这里为了不依赖外部复杂逻辑，我们做一个简单的返回，或者你需要从旧文件复制完整逻辑
        # 既然我们主要关注训练，这里如果不重要，可以返回一个占位值，
        # 但为了让 invalid_bet 统计生效，建议保留简单逻辑：
        
        # 简单的翻牌前高牌逻辑
        if not player.cards:
            return 0.0
        
        # 简单计算：(最大牌点数 - 2) / 12
        ranks = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
        values = [ranks.get(c.rank, 0) for c in player.cards]
        max_val = max(values) if values else 0
        strength = (max_val - 2) / 12.0
        
        # 如果是对子，强度加成
        if len(values) == 2 and values[0] == values[1]:
            strength += 0.2
            
        return min(1.0, strength)

    def calculate_pot_odds(self, player: Player, game: PokerGame, action_amount: Optional[int] = None) -> float:
        """计算底池赔率（用于统计日志）"""
        if action_amount is None or action_amount <= 0:
            return 0.0
        if game.pot == 0:
            return 0.0
        return game.pot / action_amount

    def is_bluffing(self, player: Player, game: PokerGame, action: str) -> bool:
        """判断是否诈唬（用于统计日志）"""
        # 定义：手牌弱 (<0.5) 且 加注 (raise)
        strength = self.calculate_hand_strength(player, game)
        return action == 'raise' and strength < 0.5