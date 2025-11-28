# state_representation.py (修复动作无效 & 智能过牌版)

import numpy as np
from typing import List, Dict, Optional
from ..game.cards import Card, RANK_VALUES
from ..game.poker_game import Player, PokerGame
from ..model.hand_evaluator import HandEvaluator

class StateRepresentation:
    """德州扑克状态表示转换器 (6-Max 专用增强版)"""
    
    def __init__(self):
        # === 维度定义 ===
        self.player_cards_dim = 52
        self.board_cards_dim = 52
        self.player_info_dim = 6 * 3
        self.game_info_dim = 5
        self.position_one_hot_dim = 6
        self.active_count_dim = 1
        
        self.total_state_dim = (
            self.player_cards_dim + 
            self.board_cards_dim + 
            self.player_info_dim + 
            self.game_info_dim +
            self.position_one_hot_dim +
            self.active_count_dim
        )
        
        self.evaluator = HandEvaluator()
        self._preflop_cache = {}

    def card_to_one_hot(self, cards: List[Card]) -> np.ndarray:
        one_hot = np.zeros(52)
        for card in cards:
            suit_idx = ['♥', '♦', '♣', '♠'].index(card.suit)
            rank_idx = list(RANK_VALUES.keys()).index(card.rank)
            position = rank_idx * 4 + suit_idx
            one_hot[position] = 1
        return one_hot
    
    def player_info_to_vector(self, players: List[Player], current_player_idx: int, game: PokerGame) -> np.ndarray:
        vector = np.zeros(self.player_info_dim)
        bb = max(game.big_blind, 1)
        
        for i in range(6):
            target_idx = (current_player_idx + i) % len(players)
            if target_idx < len(players):
                player = players[target_idx]
                base_idx = i * 3
                vector[base_idx] = min(player.stack / (bb * 100), 1.0)
                vector[base_idx + 1] = min(player.bet / (bb * 100), 1.0)
                vector[base_idx + 2] = 1.0 if player.is_active else 0.0
        return vector
    
    def get_position_vector(self, game: PokerGame, current_player: Player) -> np.ndarray:
        hero_idx = game.players.index(current_player)
        dealer_idx = game.dealer_pos
        num_players = len(game.players)
        dist_from_dealer = (hero_idx - dealer_idx + num_players) % num_players
        
        pos_vector = np.zeros(self.position_one_hot_dim)
        if dist_from_dealer < self.position_one_hot_dim:
            pos_vector[dist_from_dealer] = 1.0
        return pos_vector

    def get_active_count_vector(self, game: PokerGame) -> np.ndarray:
        active_count = sum(1 for p in game.players if p.is_active)
        normalized = (active_count - 1) / 5.0 if active_count > 0 else 0
        return np.array([normalized])

    def get_state(self, game: PokerGame, current_player: Player) -> np.ndarray:
        current_player_idx = game.players.index(current_player)
        state = np.concatenate([
            self.card_to_one_hot(current_player.cards),
            self.card_to_one_hot(game.board),
            self.player_info_to_vector(game.players, current_player_idx, game),
            self.game_info_to_vector(game),
            self.get_position_vector(game, current_player),
            self.get_active_count_vector(game)
        ])
        return state
    
    def game_info_to_vector(self, game: PokerGame) -> np.ndarray:
        bb = max(game.big_blind, 1)
        vector = np.zeros(self.game_info_dim)
        vector[0] = (game.round - 1) / 4.0
        vector[1] = min(game.pot / (bb * 100), 1.0)
        vector[2] = min(game.current_bet / (bb * 100), 1.0)
        vector[3] = game.big_blind / 100.0
        vector[4] = game.small_blind / 100.0
        return vector

    def calculate_hand_strength(self, player: Player, game: PokerGame) -> float:
        if not player.cards: return 0.0
        if not game.board: return self._quick_preflop_strength(player.cards)
        
        active_opponents = sum(1 for p in game.players if p.is_active) - 1
        win_rate = self.evaluator.calculate_win_probability(
            player.cards, game.board, 
            num_opponents=max(1, active_opponents), simulations=5
        )
        return win_rate

    def _quick_preflop_strength(self, cards: List[Card]) -> float:
        if len(cards) != 2: return 0.0
        c1, c2 = sorted(cards, key=lambda x: x.value, reverse=True)
        is_suited = c1.suit == c2.suit
        key = f"{c1.rank}{c2.rank}{'s' if is_suited else 'o'}"
        if key in self._preflop_cache: return self._preflop_cache[key]
        
        score = 0.0
        rank_scores = {14: 10, 13: 8, 12: 7, 11: 6, 10: 5}
        score += rank_scores.get(c1.value, c1.value / 2.0)
        if c1.value == c2.value:
            score *= 2
            score = max(5, score)
        if is_suited: score += 2
        gap = c1.value - c2.value - 1
        if gap == 1: score -= 1
        elif gap == 2: score -= 2
        elif gap == 3: score -= 4
        elif gap >= 4: score -= 5
        if gap <= 1 and c1.value < 12: score += 1
        
        strength = min(1.0, score / 20.0)
        self._preflop_cache[key] = strength
        return strength

    def encode_action(self, action: str, amount: Optional[int] = None, game: Optional[PokerGame] = None) -> np.ndarray:
        action_vector = np.zeros(5)
        if action == 'fold': action_vector[0] = 1
        elif action == 'call': action_vector[1] = 1
        elif action == 'raise' and game is not None:
            base_bet = max(game.current_bet, game.big_blind)
            raise_ratio = amount / base_bet if base_bet > 0 else 0
            if raise_ratio <= 2.5: action_vector[2] = 1
            elif raise_ratio <= 5.0: action_vector[3] = 1
            else: action_vector[4] = 1
        return action_vector
    
    def decode_action(self, action_vector: np.ndarray, game: PokerGame, player: Player, hand_strength: float = 0.0) -> tuple:
        """
        将神经网络输出解码为动作 (增强版：自动修正非法加注)
        """
        action_idx = np.argmax(action_vector)
        cost_to_call = max(0, game.current_bet - player.bet)
        # 基础动作定义
        # 0: Fold, 1: Call, 2: Min Raise, 3: Mid Raise, 4: All-in
        
        # === 🔥 专家规则：翻牌前全下限制 🔥 ===
        # 如果 AI 想要 All-in (action_idx == 4) 且处于翻牌前 (round == 1)
        if action_idx == 4 and game.round == 1:
            is_premium_monster = False
            # 检查是否是 AA 或 KK
            if len(player.cards) == 2:
                # 获取两张牌的点数值 (2-14)
                ranks = [c.value for c in player.cards] # 假设 Card 对象有 value 属性 (2-14)
                # 检查对子
                if ranks[0] == ranks[1]:
                    # 14=Ace, 13=King. 只有 AA 或 KK 允许
                    if ranks[0] >= 13:
                        is_premium_monster = True
            
            # 如果不是 AA/KK，禁止全下，强制降级为 Mid Raise (3) 或 Call (1)
            if not is_premium_monster:
                # 如果手牌还行 (>0.7)，降级为普通加注 (Action 3)
                # 否则降级为跟注 (Action 1)
                # 这里的 hand_strength 是外部传入的预估值
                if hand_strength > 0.7:
                    action_idx = 3 # 降级为普通加注
                else:
                    action_idx = 1 # 降级为跟注/看牌
                    
        # === 1. Fold (弃牌) ===
        if action_idx == 0:
            # 智能过牌保护：如果不需要花钱就能看牌 (Check)，那就别 Fold
            if cost_to_call == 0:
                return ('call', 0)
            return ('fold', None)

        # === 2. Call (跟注) ===
        elif action_idx == 1:
            return ('call', cost_to_call)

        # === 3. Raise (加注) ===
        # 即使 AI 选了加注，我们也要确保金额合法
        
        # 计算合法的加注范围
        # 最小加注：通常是当前下注的两倍，或者是大盲注
        min_legal_raise = max(game.current_bet * 2, game.big_blind)
        
        # 玩家最大能拿出的钱 (All-in)
        max_stack = player.bet + player.stack
        
        # 确定意向金额
        amt = 0
        if action_idx == 2: # Min Raise
            amt = min_legal_raise
        elif action_idx == 3: # Mid Raise (比如半池或 3倍大盲)
            amt = max(min_legal_raise, game.current_bet + 4 * game.big_blind)
        elif action_idx == 4: # All-in
            amt = max_stack

        # --- 🔥 关键修复逻辑：自动纠错 🔥 ---
        
        # 修正 A: 如果算出来的金额超过了身家，就只能 All-in
        if amt >= max_stack:
            amt = max_stack
        
        # 修正 B: 如果修整后的金额 < 最小加注额
        if amt < min_legal_raise:
            # 情况 B1: 如果玩家真的没钱了 (All-in 都不够最小加注)，那就允许 All-in 加注
            if amt == max_stack:
                pass # 合法，这是短筹码全下规则
            
            # 情况 B2: 玩家有钱，但 AI 算少了 -> 强制提升到最小加注
            elif max_stack >= min_legal_raise:
                amt = min_legal_raise
            
            # 情况 B3: 实在不行，就转为跟注 (Call)
            else:
                return ('call', cost_to_call)

        # 修正 C: 再次检查，如果加注额 <= 当前下注额，这其实就是跟注
        if amt <= game.current_bet:
             return ('call', cost_to_call)

        return ('raise', int(amt))