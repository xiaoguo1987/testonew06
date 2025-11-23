# 德州扑克手牌评估器

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from ..game.cards import Card, SUITS, RANKS

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('HandEvaluator')

class HandEvaluator:
    """
    德州扑克手牌评估器，用于评估牌型和计算胜率
    """
    
    # 牌型强度，从高到低
    HAND_RANKS = {
        'royal_flush': 9,
        'straight_flush': 8,
        'four_of_a_kind': 7,
        'full_house': 6,
        'flush': 5,
        'straight': 4,
        'three_of_a_kind': 3,
        'two_pair': 2,
        'one_pair': 1,
        'high_card': 0
    }
    
    def __init__(self):
        """
        初始化手牌评估器
        """
        logger.info("手牌评估器已初始化")
    
    def evaluate_hand(self, hole_cards: List[Card], community_cards: List[Card]) -> Tuple[str, List[int]]:
        """
        评估玩家手牌和公共牌的组合，返回牌型和强度值
        
        Args:
            hole_cards: 玩家的两张底牌
            community_cards: 公共牌（最多5张）
            
        Returns:
            tuple: (牌型名称, 强度值列表)
        """
        # 合并所有牌
        all_cards = hole_cards + community_cards
        
        # 确保有足够的牌进行评估
        if len(all_cards) < 5:
            return 'high_card', self._get_high_card_strength(all_cards)
        
        # 尝试各种牌型
        if self._is_royal_flush(all_cards):
            return 'royal_flush', [9]  # 皇家同花顺
        
        if self._is_straight_flush(all_cards):
            straight_high = self._get_straight_high(all_cards)
            return 'straight_flush', [8, straight_high]
        
        if self._is_four_of_a_kind(all_cards):
            four_rank = self._get_rank_counts(all_cards)[0][0]
            kicker = self._get_kickers(all_cards, [four_rank])[0]
            return 'four_of_a_kind', [7, four_rank, kicker]
        
        if self._is_full_house(all_cards):
            three_rank = self._get_rank_counts(all_cards)[0][0]
            pair_rank = self._get_rank_counts(all_cards)[1][0]
            return 'full_house', [6, three_rank, pair_rank]
        
        if self._is_flush(all_cards):
            flush_ranks = self._get_flush_ranks(all_cards)
            return 'flush', [5] + sorted(flush_ranks, reverse=True)
        
        if self._is_straight(all_cards):
            straight_high = self._get_straight_high(all_cards)
            return 'straight', [4, straight_high]
        
        if self._is_three_of_a_kind(all_cards):
            three_rank = self._get_rank_counts(all_cards)[0][0]
            kickers = self._get_kickers(all_cards, [three_rank])[:2]
            return 'three_of_a_kind', [3, three_rank] + kickers
        
        if self._is_two_pair(all_cards):
            pair_ranks = [r[0] for r in self._get_rank_counts(all_cards)[:2]]
            kicker = self._get_kickers(all_cards, pair_ranks)[0]
            return 'two_pair', [2] + sorted(pair_ranks, reverse=True) + [kicker]
        
        if self._is_one_pair(all_cards):
            pair_rank = self._get_rank_counts(all_cards)[0][0]
            kickers = self._get_kickers(all_cards, [pair_rank])[:3]
            return 'one_pair', [1, pair_rank] + kickers
        
        # 默认是高牌
        high_cards = self._get_high_card_strength(all_cards)
        return 'high_card', [0] + high_cards
    
    def _get_rank_counts(self, cards: List[Card]) -> List[Tuple[int, int]]:
        """
        获取每个牌值的出现次数，按次数降序排序
        """
        rank_counts = {}
        for card in cards:
            if card.value not in rank_counts:
                rank_counts[card.value] = 0
            rank_counts[card.value] += 1
        
        # 按次数降序，相同次数按牌值降序
        sorted_counts = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
        return sorted_counts
    
    def _is_royal_flush(self, cards: List[Card]) -> bool:
        """
        检查是否为皇家同花顺（10-J-Q-K-A同花色）
        """
        # 皇家同花顺是特殊的同花顺，所以先检查是否为同花顺
        if not self._is_straight_flush(cards):
            return False
        
        # 检查是否包含10, J, Q, K, A
        royal_ranks = [10, 11, 12, 13, 14]
        ranks = [card.value for card in cards]
        
        # 检查是否包含所有皇家牌值
        for rank in royal_ranks:
            if rank not in ranks:
                return False
        
        # 检查是否有至少5张同花色的皇家牌
        for suit in SUITS:
            suit_cards = [c for c in cards if c.suit == suit]
            suit_ranks = [c.value for c in suit_cards]
            if all(rank in suit_ranks for rank in royal_ranks):
                return True
        
        return False
    
    def _is_straight_flush(self, cards: List[Card]) -> bool:
        """
        检查是否为同花顺
        """
        # 同花顺同时满足同花和顺子的条件
        if not self._is_flush(cards):
            return False
        
        # 检查每个花色中是否有顺子
        for suit in SUITS:
            suit_cards = [c for c in cards if c.suit == suit]
            if self._is_straight(suit_cards):
                return True
        
        return False
    
    def _is_four_of_a_kind(self, cards: List[Card]) -> bool:
        """
        检查是否为四条
        """
        rank_counts = self._get_rank_counts(cards)
        return len(rank_counts) > 0 and rank_counts[0][1] == 4
    
    def _is_full_house(self, cards: List[Card]) -> bool:
        """
        检查是否为葫芦（三条+一对）
        """
        rank_counts = self._get_rank_counts(cards)
        # 需要至少两个不同的牌值，第一个出现3次，第二个出现至少2次
        return (len(rank_counts) >= 2 and 
                rank_counts[0][1] == 3 and 
                rank_counts[1][1] >= 2)
    
    def _is_flush(self, cards: List[Card]) -> bool:
        """
        检查是否为同花
        """
        # 检查每个花色的牌数
        suit_counts = {}
        for card in cards:
            if card.suit not in suit_counts:
                suit_counts[card.suit] = 0
            suit_counts[card.suit] += 1
        
        # 至少有一个花色有5张或以上的牌
        return any(count >= 5 for count in suit_counts.values())
    
    def _get_flush_ranks(self, cards: List[Card]) -> List[int]:
        """
        获取同花的牌值列表
        """
        # 找出拥有最多牌的花色
        suit_counts = {}
        for card in cards:
            if card.suit not in suit_counts:
                suit_counts[card.suit] = []
            suit_counts[card.suit].append(card.value)
        
        # 找出最大的同花牌组
        max_suit = max(suit_counts.items(), key=lambda x: len(x[1]))
        return max_suit[1]
    
    def _is_straight(self, cards: List[Card]) -> bool:
        """
        检查是否为顺子
        """
        # 获取唯一的牌值，排序
        unique_ranks = sorted(list(set([card.value for card in cards])))
        
        # 需要至少5个不同的牌值
        if len(unique_ranks) < 5:
            return False
        
        # 特殊情况：A-2-3-4-5顺子（最小顺子）
        if 14 in unique_ranks and 2 in unique_ranks and 3 in unique_ranks and 4 in unique_ranks and 5 in unique_ranks:
            return True
        
        # 检查连续的5个牌值
        for i in range(len(unique_ranks) - 4):
            if (unique_ranks[i+4] - unique_ranks[i] == 4 and
                unique_ranks[i+3] - unique_ranks[i] == 3 and
                unique_ranks[i+2] - unique_ranks[i] == 2 and
                unique_ranks[i+1] - unique_ranks[i] == 1):
                return True
        
        return False
    
    def _get_straight_high(self, cards: List[Card]) -> int:
        """
        获取顺子的最大牌值
        """
        # 获取唯一的牌值，排序
        unique_ranks = sorted(list(set([card.value for card in cards])))
        
        # 特殊情况：A-2-3-4-5顺子，最大牌值为5
        if 14 in unique_ranks and 2 in unique_ranks and 3 in unique_ranks and 4 in unique_ranks and 5 in unique_ranks:
            return 5
        
        # 检查连续的5个牌值
        for i in range(len(unique_ranks) - 4):
            if (unique_ranks[i+4] - unique_ranks[i] == 4 and
                unique_ranks[i+3] - unique_ranks[i] == 3 and
                unique_ranks[i+2] - unique_ranks[i] == 2 and
                unique_ranks[i+1] - unique_ranks[i] == 1):
                return unique_ranks[i+4]
        
        return 0  # 不是顺子
    
    def _is_three_of_a_kind(self, cards: List[Card]) -> bool:
        """
        检查是否为三条
        """
        rank_counts = self._get_rank_counts(cards)
        return len(rank_counts) > 0 and rank_counts[0][1] == 3
    
    def _is_two_pair(self, cards: List[Card]) -> bool:
        """
        检查是否为两对
        """
        rank_counts = self._get_rank_counts(cards)
        # 需要至少两个不同的牌值，都出现2次
        return len(rank_counts) >= 2 and rank_counts[0][1] == 2 and rank_counts[1][1] == 2
    
    def _is_one_pair(self, cards: List[Card]) -> bool:
        """
        检查是否为一对
        """
        rank_counts = self._get_rank_counts(cards)
        return len(rank_counts) > 0 and rank_counts[0][1] == 2
    
    def _get_high_card_strength(self, cards: List[Card]) -> List[int]:
        """
        获取高牌强度值列表
        """
        # 获取所有牌值，降序排序
        ranks = sorted([card.value for card in cards], reverse=True)
        # 取前5个最大的牌值
        return ranks[:5]
    
    def _get_kickers(self, cards: List[Card], excluded_ranks: List[int]) -> List[int]:
        """
        获取排除特定牌值后的牌值作为踢脚牌
        """
        kickers = [card.value for card in cards if card.value not in excluded_ranks]
        return sorted(kickers, reverse=True)
    
    def calculate_win_probability(self, hole_cards: List[Card], community_cards: List[Card], 
                                 num_opponents: int = 1, simulations: int = 10000) -> float:
        """
        计算获胜概率（简化版本，实际游戏中需要更复杂的蒙特卡洛模拟）
        
        Args:
            hole_cards: 玩家的两张底牌
            community_cards: 当前的公共牌
            num_opponents: 对手数量
            simulations: 模拟次数
            
        Returns:
            float: 获胜概率（0-1之间）
        """
        # 这里实现一个简化的胜率计算
        # 在实际项目中，应该使用更复杂的蒙特卡洛模拟
        #logger.info(f"计算胜率: 底牌={hole_cards}, 公共牌={community_cards}, 对手数={num_opponents}")
        
        # 基于当前牌型给出一个估计的胜率
        hand_type, _ = self.evaluate_hand(hole_cards, community_cards)
        
        # 简化的胜率估计（基于当前牌型）
        base_win_rates = {
            'royal_flush': 0.99,
            'straight_flush': 0.95,
            'four_of_a_kind': 0.90,
            'full_house': 0.85,
            'flush': 0.70,
            'straight': 0.65,
            'three_of_a_kind': 0.60,
            'two_pair': 0.45,
            'one_pair': 0.30,
            'high_card': 0.15
        }
        
        base_rate = base_win_rates.get(hand_type, 0.15)
        
        # 根据对手数量调整胜率
        opponent_penalty = 0.05 * num_opponents
        adjusted_rate = max(0.01, min(0.99, base_rate - opponent_penalty))
        
        # 根据已发的公共牌数量调整（公共牌越多，胜率越确定）
        # 翻牌前(0张公共牌)、翻牌(3张)、转牌(4张)、河牌(5张)
        community_factor = len(community_cards) / 5
        final_rate = adjusted_rate * 0.5 + adjusted_rate * 0.5 * community_factor
        
        #logger.info(f"估计胜率: {final_rate:.2%}")
        return final_rate