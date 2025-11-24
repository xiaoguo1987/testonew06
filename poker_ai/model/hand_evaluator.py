# hand_evaluator.py (优化版)

import random
import logging
from typing import List, Tuple, Dict, Optional
from ..game.cards import Card, SUITS, RANKS

class HandEvaluator:
    """德州扑克手牌评估器 (高性能版)"""
    
    def __init__(self):
        pass
    
    # --- 保持原有的 evaluate_hand 逻辑不变 (这部分很难再纯Python层面优化了) ---
    def evaluate_hand(self, hole_cards: List[Card], community_cards: List[Card]) -> Tuple[str, List[int]]:
        all_cards = hole_cards + community_cards
        if len(all_cards) < 5:
            return 'high_card', self._get_high_card_strength(all_cards)
            
        # 计数缓存，避免重复遍历
        rank_counts = {}
        suit_counts = {}
        for c in all_cards:
            rank_counts[c.value] = rank_counts.get(c.value, 0) + 1
            suit_counts[c.suit] = suit_counts.get(c.suit, 0) + 1
            
        # 快速判断同花
        flush_suit = None
        for s, count in suit_counts.items():
            if count >= 5:
                flush_suit = s
                break
                
        # 排序牌值 (从大到小)
        sorted_ranks = sorted([c.value for c in all_cards], reverse=True)
        unique_ranks = sorted(list(set(sorted_ranks)), reverse=True) # 去重
        
        # --- 牌型判断 (按概率从低到高判断，或按逻辑树) ---
        
        # 1. 同花顺 / 皇家同花顺
        if flush_suit:
            flush_cards = [c for c in all_cards if c.suit == flush_suit]
            flush_ranks = sorted([c.value for c in flush_cards], reverse=True)
            # 检查同花牌里有没有顺子
            straight_high = self._check_straight_from_ranks(flush_ranks)
            if straight_high > 0:
                if straight_high == 14 and 10 in flush_ranks: # 简单的皇家判断
                    return 'royal_flush', [9]
                return 'straight_flush', [8, straight_high]
        
        # 2. 四条
        for r, count in rank_counts.items():
            if count == 4:
                kicker = max([v for v in sorted_ranks if v != r])
                return 'four_of_a_kind', [7, r, kicker]
                
        # 3. 葫芦
        three_kind = [r for r, c in rank_counts.items() if c >= 3]
        if three_kind:
            three_kind.sort(reverse=True)
            best_three = three_kind[0]
            # 找对子 (排除掉组成三条的那三张牌)
            # 注意：可能有两组三条，另一组可以当对子用
            pairs = [r for r, c in rank_counts.items() if c >= 2 and r != best_three]
            if pairs:
                return 'full_house', [6, best_three, max(pairs)]
                
        # 4. 同花
        if flush_suit:
            # flush_cards 已经在上面计算过了，这里为了逻辑独立重新筛选一下或者缓存
            flush_cards = [c for c in all_cards if c.suit == flush_suit]
            return 'flush', [5] + sorted([c.value for c in flush_cards], reverse=True)[:5]
            
        # 5. 顺子
        straight_high = self._check_straight_from_ranks(unique_ranks)
        if straight_high > 0:
            return 'straight', [4, straight_high]
            
        # 6. 三条
        if three_kind: # 上面葫芦逻辑里算过
             # 找踢脚
             kickers = [v for v in sorted_ranks if v != three_kind[0]][:2]
             return 'three_of_a_kind', [3, three_kind[0]] + kickers
             
        # 7. 两对
        pairs = [r for r, c in rank_counts.items() if c >= 2]
        if len(pairs) >= 2:
            pairs.sort(reverse=True)
            kicker = max([v for v in sorted_ranks if v != pairs[0] and v != pairs[1]])
            return 'two_pair', [2, pairs[0], pairs[1], kicker]
            
        # 8. 一对
        if pairs:
            kicker_list = [v for v in sorted_ranks if v != pairs[0]][:3]
            return 'one_pair', [1, pairs[0]] + kicker_list
            
        return 'high_card', [0] + sorted_ranks[:5]

    def _check_straight_from_ranks(self, unique_ranks: List[int]) -> int:
        """输入去重后的倒序rank列表，返回最大顺子头，无则返回0"""
        if len(unique_ranks) < 5: return 0
        # A-5 特殊处理: A(14), 5, 4, 3, 2
        if 14 in unique_ranks and 2 in unique_ranks and 3 in unique_ranks and 4 in unique_ranks and 5 in unique_ranks:
            # 检查有没有更大的顺子，比如 6,5,4,3,2。如果没有，暂且认为A-5存在
            # 但这里逻辑是：先找普通顺子，找不到再返回5
            pass 

        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i] - unique_ranks[i+4] == 4:
                return unique_ranks[i]
        
        if 14 in unique_ranks and 5 in unique_ranks and 4 in unique_ranks and 3 in unique_ranks and 2 in unique_ranks:
            return 5
            
        return 0

    def _get_high_card_strength(self, cards: List[Card]) -> List[int]:
        return sorted([c.value for c in cards], reverse=True)[:5]

    # ==========================================
    # 🔥 核心优化：获胜概率计算 🔥
    # ==========================================
    def calculate_win_probability(self, hole_cards: List[Card], community_cards: List[Card], 
                                 num_opponents: int = 1, simulations: int = 5) -> float:
        """
        计算获胜概率 (高性能版)
        Args:
            simulations: 默认值极低，专为特征提取优化
        """
        # 1. 创建一个"虚拟"牌堆 (只包含索引，不创建对象)
        # 假设 Deck 是 0-51 的整数列表
        # 0-12: Hearts 2-A, 13-25: Diamonds...
        # 这种映射需要和 Card 类一致，这里为了兼容你的代码，还是用 Card 对象操作
        # 但我们只在循环外创建一次 full deck
        
        # 找出已知的牌
        known_cards_set = set()
        for c in hole_cards + community_cards:
            known_cards_set.add(f"{c.rank}{c.suit}")
            
        # 构建剩余牌堆
        deck_rest = []
        for r in ['2','3','4','5','6','7','8','9','T','J','Q','K','A']:
            for s in ['♥', '♦', '♣', '♠']:
                if f"{r}{s}" not in known_cards_set:
                    deck_rest.append(Card(r, s))
                    
        wins = 0
        needed_community = 5 - len(community_cards)
        
        for _ in range(simulations):
            random.shuffle(deck_rest)
            current_idx = 0
            
            # 补全公共牌
            sim_community = list(community_cards)
            for _ in range(needed_community):
                sim_community.append(deck_rest[current_idx])
                current_idx += 1
                
            # 发对手牌
            opp_hands = []
            for _ in range(num_opponents):
                h = [deck_rest[current_idx], deck_rest[current_idx+1]]
                current_idx += 2
                opp_hands.append(h)
                
            # 评估
            # 这里必须调用 self.evaluate_hand 
            # (虽然它有点慢，但比原来的实现要快，因为不用重造对象)
            _, my_score = self.evaluate_hand(hole_cards, sim_community)
            
            won = True
            for opp in opp_hands:
                _, opp_score = self.evaluate_hand(opp, sim_community)
                if opp_score > my_score: # 列表比较
                    won = False
                    break
                elif opp_score == my_score:
                    # 平局算输或者算0.5，这里为了保守起见算输
                    won = False 
                    break
            
            if won:
                wins += 1
                
        return wins / simulations