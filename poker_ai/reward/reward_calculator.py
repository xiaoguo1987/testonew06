import numpy as np
import random
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Set
from ..game.poker_game import PokerGame, Player
from ..game.cards import Card, evaluate_hand, compare_hands

class RewardCalculator:
    """德州扑克奖励计算器（精细化优化版）"""
    
    def __init__(self):
        # 定义不同奖励类型的权重（新增无效下注惩罚）
        self.reward_weights = {
            'winning_hand': 1.0,          # 赢牌奖励
            'losing_hand': -0.8,          # 输牌惩罚
            'pot_odds': 0.2,              # 底池赔率奖励
            'hand_strength': 0.3,         # 手牌强度奖励
            'bluff_success': 0.4,         # 成功诈唬奖励
            'fold_when_weak': 0.3,        # 及时弃牌奖励
            'aggressive_play': 0.1,       # 主动进攻奖励
            'position_advantage': 0.15,   # 位置优势奖励
            'bankroll_change': 0.5,       # 筹码变化奖励
            'decision_consistency': 0.2,  # 决策一致性奖励
            'position_utilization': 0.2,  # 位置利用奖励
            'invalid_bet': 0.4            # 无效下注惩罚
        }
        
        # 定义不同游戏阶段的奖励调整因子
        self.stage_factors = {
            1: 1.0,  # 翻牌前
            2: 1.2,  # 翻牌
            3: 1.5,  # 转牌
            4: 2.0   # 河牌
        }
        
        # 蒙特卡洛模拟参数
        self.monte_carlo_iterations = 1000  # 模拟次数
        self.player_last_state = {}  # 存储玩家上一轮状态（用于一致性判断）

    def calculate_hand_strength(self, player: Player, game: PokerGame) -> float:
        """估算手牌强度（0-1之间），含蒙特卡洛缓存优化"""
        if not player.is_active:
            return 0.0
        
        # 缓存键：手牌+公共牌（确保可哈希）
        cache_key = self._get_hand_cache_key(player.cards, game.board)
        
        # 优先从缓存获取，无缓存则执行模拟
        if game.board:
            win_prob = self._cached_win_probability(cache_key, player, game)
        else:
            win_prob = self._preflop_hand_strength(player.cards)
        
        # 记录当前手牌强度（用于决策一致性判断）
        self._update_player_last_state(player, win_prob)
        return win_prob

    def _get_hand_cache_key(self, player_cards: List[Card], board: List[Card]) -> str:
        """生成手牌+公共牌的缓存键（可哈希）"""
        hand_str = ",".join(sorted([f"{c.rank}-{c.suit}" for c in player_cards]))
        board_str = ",".join(sorted([f"{c.rank}-{c.suit}" for c in board]))
        return f"{hand_str}|{board_str}"

    @lru_cache(maxsize=10000)  # 缓存10000个结果，自动淘汰旧数据
    def _cached_win_probability(self, cache_key: str, player: Player, game: PokerGame) -> float:
        """缓存版胜率计算（核心蒙特卡洛逻辑）"""
        return self._estimate_win_probability(player, game)

    def _preflop_hand_strength(self, cards: List[Card]) -> float:
        """翻牌前的手牌强度评估"""
        if len(cards) != 2:
            return 0.0
        
        rank1, rank2 = cards[0].rank, cards[1].rank
        is_suited = cards[0].suit == cards[1].suit
        key = tuple(sorted([rank1, rank2], reverse=True))
        
        # 常见起手牌强度映射
        hand_strength_map = {
            ('A', 'A'): 0.95, ('K', 'K'): 0.85, ('Q', 'Q'): 0.75, ('J', 'J'): 0.65, ('T', 'T'): 0.55,
            ('9', '9'): 0.45, ('8', '8'): 0.35, ('A', 'K'): 0.8 if is_suited else 0.65,
            ('A', 'Q'): 0.75 if is_suited else 0.6, ('A', 'J'): 0.7 if is_suited else 0.55,
            ('A', 'T'): 0.65 if is_suited else 0.5, ('K', 'Q'): 0.65 if is_suited else 0.5,
            ('K', 'J'): 0.6 if is_suited else 0.45, ('Q', 'J'): 0.55 if is_suited else 0.4,
            ('J', 'T'): 0.5 if is_suited else 0.35, ('T', '9'): 0.45 if is_suited else 0.3
        }
        
        base_strength = hand_strength_map.get(key, 0.3)
        # 同色/连续牌额外加成
        if is_suited and key not in hand_strength_map:
            base_strength += 0.1
        if abs(self._rank_to_value(rank1) - self._rank_to_value(rank2)) == 1:
            base_strength += 0.1
        
        return min(1.0, max(0.0, base_strength))

    def _rank_to_value(self, rank: str) -> int:
        """将牌点转换为数值"""
        rank_values = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
        return rank_values.get(rank, 0)

    def _get_used_cards(self, player: Player, game: PokerGame) -> Set[Card]:
        """获取已使用的牌（玩家手牌+公共牌+其他玩家亮牌）"""
        used_cards = set(player.cards)
        used_cards.update(game.board)
        for p in game.players:
            if p != player and hasattr(p, 'revealed_cards'):
                used_cards.update(p.revealed_cards)
        return used_cards

    def _create_remaining_deck(self, used_cards: Set[Card]) -> List[Card]:
        """创建剩余牌库（排除已使用的牌）"""
        suits = ['♥', '♦', '♣', '♠']
        ranks = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
        full_deck = [Card(rank, suit) for suit in suits for rank in ranks]
        return [card for card in full_deck if card not in used_cards]

    def _estimate_win_probability(self, player: Player, game: PokerGame) -> float:
        """蒙特卡洛模拟估算胜率（核心逻辑）"""
        opponents = [p for p in game.players if p != player and p.is_active]
        if not opponents:
            return 1.0
        
        current_board_size = len(game.board)
        remaining_community_needed = 5 - current_board_size
        used_cards = self._get_used_cards(player, game)
        initial_deck = self._create_remaining_deck(used_cards)
        
        # 牌库不足时返回简化计算
        min_required = remaining_community_needed + 2 * len(opponents)
        if len(initial_deck) < min_required:
            return 0.5 + (current_board_size / 5) * 0.3
        
        # 模拟统计胜率
        wins = 0
        for _ in range(self.monte_carlo_iterations):
            deck = initial_deck.copy()
            random.shuffle(deck)
            
            # 抽取剩余公共牌
            community_cards = deck[:remaining_community_needed]
            del deck[:remaining_community_needed]
            full_board = game.board + community_cards
            
            # 为对手分配手牌
            opponent_hands = []
            for _ in opponents:
                if len(deck) < 2:
                    break
                opponent_hands.append([deck.pop(), deck.pop()])
            if len(opponent_hands) != len(opponents):
                continue
            
            # 评估所有玩家手牌
            player_hand_rank = evaluate_hand(player.cards + full_board)
            opponent_ranks = [evaluate_hand(h + full_board) for h in opponent_hands]
            
            # 判断是否获胜
            is_winner = all(compare_hands(opp_rank, player_hand_rank) < 0 for opp_rank in opponent_ranks)
            if is_winner:
                wins += 1
        
        return min(1.0, max(0.0, wins / self.monte_carlo_iterations))

    def _update_player_last_state(self, player: Player, current_strength: float) -> None:
        """更新玩家上一轮状态（手牌强度+动作）"""
        if player.name not in self.player_last_state:
            self.player_last_state[player.name] = {
                'hand_strength': current_strength,
                'action': None
            }
        else:
            self.player_last_state[player.name]['hand_strength'] = current_strength

    def _check_decision_consistency(self, player: Player, action: str, game: PokerGame) -> float:
        """检查决策一致性，返回奖励/惩罚值"""
        if player.name not in self.player_last_state:
            return 0.0
        
        last_state = self.player_last_state[player.name]
        last_action = last_state.get('action')
        last_strength = last_state.get('hand_strength', 0.0)
        current_strength = self.calculate_hand_strength(player, game)
        strength_change = current_strength - last_strength

        # 矛盾场景：前一轮进攻（raise/call）→ 当前弃牌，且强度下降<0.2
        if last_action in ['raise', 'call'] and action == 'fold' and strength_change > -0.2:
            consistency_reward = -self.reward_weights['decision_consistency']
        # 一致场景：前一轮弃牌→当前弃牌，或前一轮进攻→当前进攻（强度无大幅下降）
        elif (last_action == 'fold' and action == 'fold') or \
             (last_action in ['raise', 'call'] and action in ['raise', 'call'] and strength_change > -0.3):
            consistency_reward = self.reward_weights['decision_consistency'] * 0.5
        else:
            consistency_reward = 0.0
        
        # 更新当前动作到历史状态
        self.player_last_state[player.name]['action'] = action
        return consistency_reward

    def calculate_position_advantage(self, player: Player, game: PokerGame, action: str) -> float:
        """计算位置优势+位置利用奖励（动作联动）"""
        position = player.position
        dealer_pos = game.dealer_pos
        relative_pos = (position - dealer_pos) % 6  # 相对庄家位置

        # 基础位置奖励
        base_rewards = {
            0: 0.8,  # 庄家（后位）
            1: -0.5, # 小盲（前位）
            2: -0.2, # 大盲（前位）
            3: 0.1,  # 枪口+1（中位）
            4: 0.4,  # 中位
            5: 0.6   # 枪口（前位）
        }
        base_reward = base_rewards.get(relative_pos, 0.0)

        # 位置利用奖励：后位（庄家/枪口）主动进攻额外加成
        if relative_pos in [0, 5] and action == 'raise':
            base_reward += self.reward_weights['position_utilization'] * 0.8
        # 前位（小盲/大盲）盲目进攻额外惩罚
        elif relative_pos in [1, 2] and action == 'raise' and self.calculate_hand_strength(player, game) < 0.4:
            base_reward -= self.reward_weights['position_utilization'] * 0.5

        return base_reward

    def calculate_pot_odds(self, player: Player, game: PokerGame, action_amount: Optional[int] = None) -> float:
        """计算底池赔率奖励"""
        if action_amount is None or action_amount <= 0:
            return 0.0
        
        pot_size = game.pot
        odds = pot_size / action_amount if action_amount > 0 else 0
        if odds > 3.0:
            return 1.0
        elif odds > 1.5:
            return 0.5
        elif odds > 1.0:
            return 0.0
        else:
            return -0.5

    def is_bluffing(self, player: Player, game: PokerGame, action: str) -> bool:
        """判断玩家是否在诈唬"""
        hand_strength = self.calculate_hand_strength(player, game)
        return hand_strength < 0.3 and action == 'raise'

    def calculate_reward(self, player: Player, game: PokerGame, action: str, 
                         action_amount: Optional[int] = None, 
                         is_winner: bool = False, 
                         profit_loss: float = 0.0,
                         done: bool = False) -> float:
        """计算综合奖励（含所有精细化因子）"""
        total_reward = 0.0
        hand_strength = self.calculate_hand_strength(player, game)

        # 1. 筹码变化奖励
        bankroll_reward = profit_loss / (game.big_blind * 10)  # 归一化
        total_reward += self.reward_weights['bankroll_change'] * bankroll_reward

        # 2. 赢牌/输牌奖励
        if is_winner:
            total_reward += self.reward_weights['winning_hand']
        else:
            active_players = [p for p in game.players if p.is_active]
            if player.is_active and len(active_players) > 1:
                total_reward += self.reward_weights['losing_hand']

        # 3. 手牌强度奖励
        total_reward += self.reward_weights['hand_strength'] * hand_strength

        # 4. 底池赔率奖励
        if action in ['call', 'raise'] and action_amount is not None:
            pot_odds_reward = self.calculate_pot_odds(player, game, action_amount)
            total_reward += self.reward_weights['pot_odds'] * pot_odds_reward

        # 5. 位置优势+利用奖励
        position_reward = self.calculate_position_advantage(player, game, action)
        total_reward += position_reward

        # 6. 主动进攻奖励
        if action == 'raise':
            total_reward += self.reward_weights['aggressive_play']

        # 7. 及时弃牌奖励
        if action == 'fold' and hand_strength < 0.2:
            total_reward += self.reward_weights['fold_when_weak']

        # 8. 成功诈唬奖励
        if self.is_bluffing(player, game, action) and is_winner:
            total_reward += self.reward_weights['bluff_success']

        # 9. 决策一致性奖励
        consistency_reward = self._check_decision_consistency(player, action, game)
        total_reward += consistency_reward

        # 10. 无效下注惩罚（手牌弱+赔率差+大额加注）
        if action == 'raise' and action_amount is not None:
            pot_odds = self.calculate_pot_odds(player, game, action_amount)
            if hand_strength < 0.3 and pot_odds < 0.0 and action_amount > game.pot * 0.5:
                total_reward -= self.reward_weights['invalid_bet']

        # 动态阶段因子：手牌越弱，阶段权重越高（鼓励谨慎）
        stage_factor = self.stage_factors.get(game.round, 1.0)
        stage_factor *= (1.2 - hand_strength * 0.5)  # 范围：0.7~1.2
        total_reward *= stage_factor

        # 重置玩家历史状态（局结束时）
        if done:
            self.player_last_state.pop(player.name, None)

        # 限制奖励范围
        return max(-2.0, min(2.0, total_reward))

    def calculate_episode_rewards(self, player: Player, game: PokerGame, 
                                action_history: List[Tuple[str, Optional[int]]],
                                is_winner: bool, profit_loss: float) -> List[float]:
        """计算一整局游戏中每个动作的奖励"""
        rewards = []
        for idx, (action, amount) in enumerate(action_history):
            done = idx == len(action_history) - 1  # 最后一个动作标记为局结束
            immediate_reward = self.calculate_reward(
                player, game, action, amount, is_winner=False, profit_loss=0.0, done=done
            )
            rewards.append(immediate_reward)
        
        # 最后动作叠加最终奖励（赢牌/筹码变化）
        if rewards:
            final_reward = self.calculate_reward(
                player, game, action_history[-1][0], action_history[-1][1],
                is_winner, profit_loss, done=True
            )
            rewards[-1] += final_reward

        return rewards


class AdaptiveRewardCalculator:
    """自适应奖励计算器（精细化调整逻辑）"""
    
    def __init__(self, base_calculator: Optional[RewardCalculator] = None):
        self.base_calculator = base_calculator or RewardCalculator()
        # 添加reward_weights属性引用，解决AttributeError问题
        self.reward_weights = self.base_calculator.reward_weights
        self.performance_history = []  # 存储最近表现（含策略指标）
        self.adjustment_factor = 0.05  # 调整因子
        self.history_size = 50  # 历史记录大小

    def record_performance(self, profit_loss: float, hands_played: int, 
                         bluff_rate: float, invalid_bet_rate: float) -> None:
        """记录AI表现（新增诈唬率、无效下注率）"""
        profit_per_hand = profit_loss / hands_played if hands_played > 0 else 0
        self.performance_history.append({
            'profit_per_hand': profit_per_hand,
            'bluff_rate': bluff_rate,
            'invalid_bet_rate': invalid_bet_rate
        })
        
        # 保持历史记录大小
        if len(self.performance_history) > self.history_size:
            self.performance_history.pop(0)

    def adjust_rewards(self) -> None:
        """根据多维度表现动态调整奖励权重"""
        if len(self.performance_history) < 10:
            return  # 数据不足，不调整
        
        # 提取关键指标
        profits = [p['profit_per_hand'] for p in self.performance_history]
        bluff_rates = [p['bluff_rate'] for p in self.performance_history]
        invalid_bet_rates = [p['invalid_bet_rate'] for p in self.performance_history]
        
        # 计算统计值
        recent_avg_profit = np.mean(profits[-5:])
        overall_avg_profit = np.mean(profits)
        bluff_std = np.std(bluff_rates)  # 诈唬率波动（策略多样性）
        avg_invalid_bet = np.mean(invalid_bet_rates)  # 平均无效下注率

        # 场景1：收益下降且策略单一 → 鼓励探索
        if recent_avg_profit < overall_avg_profit * 0.8 and bluff_std < 0.05:
            self.base_calculator.reward_weights['aggressive_play'] += self.adjustment_factor
            self.base_calculator.reward_weights['bluff_success'] += self.adjustment_factor * 0.5
            if self.base_calculator.reward_weights['fold_when_weak'] > 0.1:
                self.base_calculator.reward_weights['fold_when_weak'] -= self.adjustment_factor * 0.5

        # 场景2：收益稳定但无效下注多 → 惩罚无效决策
        elif recent_avg_profit > overall_avg_profit * 0.9 and avg_invalid_bet > 0.15:
            self.base_calculator.reward_weights['invalid_bet'] += self.adjustment_factor * 0.5
            self.base_calculator.reward_weights['fold_when_weak'] += self.adjustment_factor * 0.3

        # 场景3：诈唬过度（诈唬率>30%）→ 降低诈唬奖励
        if np.mean(bluff_rates) > 0.3:
            self.base_calculator.reward_weights['bluff_success'] -= self.adjustment_factor * 0.8
            self.base_calculator.reward_weights['hand_strength'] += self.adjustment_factor * 0.5

        # 限制权重范围（0~2.0）
        for key in self.base_calculator.reward_weights:
            self.base_calculator.reward_weights[key] = max(0.0, min(2.0, self.base_calculator.reward_weights[key]))

    def calculate_hand_strength(self, player: Player, game: PokerGame) -> float:
        """估算手牌强度（代理到基础计算器）"""
        return self.base_calculator.calculate_hand_strength(player, game)
    
    def calculate_pot_odds(self, player: Player, game: PokerGame, action_amount: Optional[int] = None) -> float:
        """计算底池赔率（代理到基础计算器）"""
        return self.base_calculator.calculate_pot_odds(player, game, action_amount)
    
    def is_bluffing(self, player: Player, game: PokerGame, action: str) -> bool:
        """判断玩家是否在诈唬（代理到基础计算器）"""
        return self.base_calculator.is_bluffing(player, game, action)
    
    def calculate_reward(self, *args, **kwargs) -> float:
        """兼容基础计算器接口"""
        return self.base_calculator.calculate_reward(*args, **kwargs)


# 示例使用
if __name__ == "__main__":
    from ..game.poker_game import Player, PokerGame
    from ..game.cards import Card
    
    # 创建玩家和游戏
    players = [Player(f"Player {i}", 10000, i > 0) for i in range(6)]
    game = PokerGame(5, 10, players)
    game.start_new_hand()
    game.board = [Card('5', '♥'), Card('10', '♣'), Card('K', '♠')]  # 模拟翻牌
    
    # 初始化计算器
    reward_calc = RewardCalculator()
    player = players[0]
    
    # 测试核心功能
    print(f"玩家手牌: {[f'{c.rank}{c.suit[0]}' for c in player.cards]}")
    print(f"公共牌: {[f'{c.rank}{c.suit[0]}' for c in game.board]}")
    print(f"手牌强度: {reward_calc.calculate_hand_strength(player, game):.4f}")
    
    # 测试奖励计算
    reward = reward_calc.calculate_reward(
        player, game, action='raise', action_amount=50, is_winner=False, profit_loss=-50
    )
    print(f"奖励值: {reward:.2f}")
    
    # 测试自适应调整
    adaptive_calc = AdaptiveRewardCalculator()
    for _ in range(10):
        adaptive_calc.record_performance(profit_loss=-100, hands_played=20, bluff_rate=0.4, invalid_bet_rate=0.2)
    adaptive_calc.adjust_rewards()
    print("调整后奖励权重:")
    for k, v in adaptive_calc.base_calculator.reward_weights.items():
        print(f"  {k}: {v:.2f}")