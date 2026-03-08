# 德州扑克卡片和牌型评估模块

import random
from typing import List, Tuple, Optional

# 定义扑克牌的花色和点数
SUITS = ['♥', '♦', '♣', '♠']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
RANK_VALUES = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

class Card:
    """表示一张扑克牌"""
    def __init__(self, rank: str, suit: str):
        if rank not in RANKS:
            raise ValueError(f"无效的点数: {rank}")
        if suit not in SUITS:
            raise ValueError(f"无效的花色: {suit}")
        self.rank = rank
        self.suit = suit
        self.value = RANK_VALUES[rank]
    
    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self) -> int:
        # 确保Card对象可以被哈希化，用于集合和字典键
        return hash((self.rank, self.suit))


_CARD_CACHE = tuple(Card(rank, suit) for rank in RANKS for suit in SUITS)
_CARD_CODE_LOOKUP = {
    (card.rank, card.suit): idx for idx, card in enumerate(_CARD_CACHE)
}


def card_to_code(card: Card) -> int:
    """将 Card 映射到稳定的整数编码（rank-major, suit-minor）。"""
    return _CARD_CODE_LOOKUP[(card.rank, card.suit)]


def code_to_card(code: int) -> Card:
    """将整数编码映射回共享 Card 实例。"""
    return _CARD_CACHE[int(code)]

class Deck:
    """表示一副扑克牌"""
    def __init__(self):
        self.cards = list(range(len(_CARD_CACHE)))
        self.shuffle()
    
    def shuffle(self) -> None:
        """洗牌"""
        random.shuffle(self.cards)
    
    def deal(self) -> Card:
        """发一张牌"""
        if not self.cards:
            raise ValueError("牌堆已空")
        return code_to_card(self.cards.pop())

    def deal_code(self) -> int:
        """直接发整数编码，供高性能环境路径使用。"""
        if not self.cards:
            raise ValueError("牌堆已空")
        return int(self.cards.pop())
    
    def __len__(self) -> int:
        return len(self.cards)

# 牌型等级常量
HAND_RANKS = {
    'high_card': 0,
    'one_pair': 1,
    'two_pair': 2,
    'three_of_a_kind': 3,
    'straight': 4,
    'flush': 5,
    'full_house': 6,
    'four_of_a_kind': 7,
    'straight_flush': 8,
    'royal_flush': 9
}

def evaluate_hand(cards: List[Card]) -> Tuple[int, List[int]]:
    """
    评估手牌的牌型和强度
    返回: (牌型等级, 强度值列表)
    """
    # 确保我们使用5张牌进行评估
    if len(cards) < 5:
        raise ValueError("需要至少5张牌来评估手牌")
    
    # 按点数分组并排序
    rank_counts = {}
    for card in cards:
        rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
    
    # 检查同花顺、顺子和同花
    is_flush = all(card.suit == cards[0].suit for card in cards)
    sorted_ranks = sorted([card.value for card in cards], reverse=True)
    
    # 检查顺子（考虑A-5顺子的特殊情况）
    is_straight = False
    if len(set(sorted_ranks)) == 5:
        if sorted_ranks[0] - sorted_ranks[4] == 4:
            is_straight = True
        # 特殊情况：A-5顺子 (A,2,3,4,5)
        elif set(sorted_ranks) == {14, 2, 3, 4, 5}:
            is_straight = True
            # 调整A-5顺子的排序，使5为最大牌
            sorted_ranks = [5, 4, 3, 2, 14]
    
    # 检查皇家同花顺
    if is_straight and is_flush and sorted_ranks[0] == 14:
        return (HAND_RANKS['royal_flush'], [])
    
    # 检查同花顺
    if is_straight and is_flush:
        return (HAND_RANKS['straight_flush'], sorted_ranks)
    
    # 检查四条
    if 4 in rank_counts.values():
        four_rank = [rank for rank, count in rank_counts.items() if count == 4][0]
        kicker = sorted([card.value for card in cards if card.rank != four_rank], reverse=True)
        return (HAND_RANKS['four_of_a_kind'], [RANK_VALUES[four_rank]] + kicker)
    
    # 检查葫芦
    if 3 in rank_counts.values() and 2 in rank_counts.values():
        three_rank = [rank for rank, count in rank_counts.items() if count == 3][0]
        pair_rank = [rank for rank, count in rank_counts.items() if count == 2][0]
        return (HAND_RANKS['full_house'], [RANK_VALUES[three_rank], RANK_VALUES[pair_rank]])
    
    # 检查同花
    if is_flush:
        return (HAND_RANKS['flush'], sorted_ranks)
    
    # 检查顺子
    if is_straight:
        return (HAND_RANKS['straight'], sorted_ranks)
    
    # 检查三条
    if 3 in rank_counts.values():
        three_rank = [rank for rank, count in rank_counts.items() if count == 3][0]
        kickers = sorted([card.value for card in cards if card.rank != three_rank], reverse=True)
        return (HAND_RANKS['three_of_a_kind'], [RANK_VALUES[three_rank]] + kickers)
    
    # 检查两对
    pair_ranks = [rank for rank, count in rank_counts.items() if count == 2]
    if len(pair_ranks) == 2:
        pair_ranks_values = sorted([RANK_VALUES[rank] for rank in pair_ranks], reverse=True)
        kicker = [card.value for card in cards if card.rank not in pair_ranks][0]
        return (HAND_RANKS['two_pair'], pair_ranks_values + [kicker])
    
    # 检查一对
    if len(pair_ranks) == 1:
        pair_rank_value = RANK_VALUES[pair_ranks[0]]
        kickers = sorted([card.value for card in cards if card.rank != pair_ranks[0]], reverse=True)
        return (HAND_RANKS['one_pair'], [pair_rank_value] + kickers)
    
    # 高牌
    return (HAND_RANKS['high_card'], sorted_ranks)

def compare_hands(hand1: List[Card], hand2: List[Card]) -> int:
    """
    比较两手牌的强弱
    返回: 1 (hand1赢), 0 (平局), -1 (hand2赢)
    """
    # 从7张牌中选出最佳的5张牌组合
    def get_best_hand(seven_cards: List[Card]) -> Tuple[int, List[int]]:
        import itertools
        best_rank = -1
        best_strength = []
        for combination in itertools.combinations(seven_cards, 5):
            rank, strength = evaluate_hand(list(combination))
            if rank > best_rank or (rank == best_rank and strength > best_strength):
                best_rank = rank
                best_strength = strength
        return (best_rank, best_strength)
    
    rank1, strength1 = get_best_hand(hand1)
    rank2, strength2 = get_best_hand(hand2)
    
    if rank1 > rank2:
        return 1
    elif rank1 < rank2:
        return -1
    else:
        # 牌型相同，比较强度值
        for s1, s2 in zip(strength1, strength2):
            if s1 > s2:
                return 1
            elif s1 < s2:
                return -1
        return 0  # 平局