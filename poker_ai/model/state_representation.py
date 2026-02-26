# state_representation.py (6-Max 专用增强版)

import numpy as np
from typing import List, Dict, Optional
from ..game.cards import Card, RANK_VALUES
from ..game.poker_game import Player, PokerGame
from ..model.hand_evaluator import HandEvaluator

class StateRepresentation:
    """德州扑克状态表示转换器 (6-Max 专用增强版)"""
    
    def __init__(self):
        # === 维度定义 ===
        # 1. 玩家手牌 (13*4 = 52)
        self.player_cards_dim = 52
        
        # 2. 公共牌 (13*4 = 52)
        self.board_cards_dim = 52
        
        # 3. 所有玩家详细信息 (6个玩家 * 3个特征: Stack, Bet, Active) = 18
        self.player_info_dim = 6 * 3
        
        # 4. 游戏全局信息 (Round, Pot, Bet, BB, SB) = 5
        self.game_info_dim = 5
        
        # 5. [新增] Hero绝对位置 One-Hot (BTN, SB, BB, UTG, MP, CO) = 6维
        self.position_one_hot_dim = 6
        
        # 6. [新增] 活跃玩家数量 (归一化数值) = 1维
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
        # 预计算索引映射，避免在高频路径中反复 list.index
        self._suit_to_idx = {'♥': 0, '♦': 1, '♣': 2, '♠': 3}
        self._rank_to_idx = {rank: idx for idx, rank in enumerate(RANK_VALUES.keys())}

    def card_to_one_hot(self, cards: List[Card]) -> np.ndarray:
        """将牌转换为one-hot编码"""
        one_hot = np.zeros(52, dtype=np.float32)
        for card in cards:
            suit_idx = self._suit_to_idx[card.suit]
            rank_idx = self._rank_to_idx[card.rank]
            position = rank_idx * 4 + suit_idx
            one_hot[position] = 1
        return one_hot
    
    def player_info_to_vector(self, players: List[Player], current_player_idx: int, game: PokerGame) -> np.ndarray:
        """玩家状态向量，保持相对位置顺序，让AI总是在第一个位置看到自己"""
        vector = np.zeros(self.player_info_dim)
        bb = max(game.big_blind, 1)
        
        for i in range(6): # 固定循环6次，确保维度固定
            # 这里的逻辑是：将当前玩家(Hero)放在向量的开头(i=0)，其他人顺时针排列
            # 这样 AI 永远知道 vector[0:3] 是自己的信息
            target_idx = (current_player_idx + i) % len(players)
            
            if target_idx < len(players):
                player = players[target_idx]
                base_idx = i * 3
                
                # 归一化特征
                stack_normalized = min(player.stack / (bb * 100), 1.0)
                bet_normalized = min(player.bet / (bb * 100), 1.0)
                is_active = 1.0 if player.is_active else 0.0
                
                vector[base_idx] = stack_normalized
                vector[base_idx + 1] = bet_normalized
                vector[base_idx + 2] = is_active
                
        return vector
    
    def get_position_vector(self, game: PokerGame, current_player: Player) -> np.ndarray:
        """
        [新增] 获取 Hero 的绝对位置 One-Hot 向量
        6-Max 位置定义相对于庄家 (Dealer):
        0: BTN (Dealer)
        1: SB
        2: BB
        3: UTG (Under the Gun)
        4: MP (Middle Position / Hijack)
        5: CO (Cutoff)
        """
        hero_idx = game.players.index(current_player)
        dealer_idx = game.dealer_pos
        num_players = len(game.players) # 应该是6
        
        # 计算相对于庄家的位置距离
        # 距离 0 = BTN, 1 = SB, 2 = BB ...
        dist_from_dealer = (hero_idx - dealer_idx + num_players) % num_players
        
        # 创建 One-Hot
        pos_vector = np.zeros(self.position_one_hot_dim)
        # 保护一下防止 num_players 变动导致越界
        if dist_from_dealer < self.position_one_hot_dim:
            pos_vector[dist_from_dealer] = 1.0
            
        return pos_vector

    def get_active_count_vector(self, game: PokerGame) -> np.ndarray:
        """[新增] 获取活跃玩家数量特征"""
        active_count = sum(1 for p in game.players if p.is_active)
        # 归一化：(活跃人数 - 1) / 5.0。 
        # 结果范围 0.0 (Heads-up) 到 1.0 (6人全在)
        normalized = (active_count - 1) / 5.0 if active_count > 0 else 0
        return np.array([normalized])

    def get_state(self, game: PokerGame, current_player: Player) -> np.ndarray:
        """获取完整的游戏状态向量"""
        current_player_idx = game.players.index(current_player)
        
        # 组合所有特征
        state = np.concatenate([
            # 1. 自身手牌
            self.card_to_one_hot(current_player.cards),
            # 2. 公共牌
            self.card_to_one_hot(game.board),
            # 3. 所有玩家状态 (Hero排在最前)
            self.player_info_to_vector(game.players, current_player_idx, game),
            # 4. 游戏全局信息
            self.game_info_to_vector(game),
            # 5. [新增] 绝对位置信息 (至关重要)
            self.get_position_vector(game, current_player),
            # 6. [新增] 活跃人数信息
            self.get_active_count_vector(game)
        ])
        return state
    
    def game_info_to_vector(self, game: PokerGame) -> np.ndarray:
        # 保持之前的优化逻辑
        bb = max(game.big_blind, 1)
        vector = np.zeros(self.game_info_dim)
        vector[0] = (game.round - 1) / 4.0
        vector[1] = min(game.pot / (bb * 100), 1.0)
        vector[2] = min(game.current_bet / (bb * 100), 1.0)
        vector[3] = game.big_blind / 100.0
        vector[4] = game.small_blind / 100.0
        return vector

    # ==========================================
    # 保持之前的性能优化 (calculate_hand_strength 等)
    # ==========================================
    def calculate_hand_strength(self, player: Player, game: PokerGame) -> float:
        # 直接复用刚才那个高性能版本的代码
        if not player.cards: return 0.0
        if not game.board: return self._quick_preflop_strength(player.cards)
        
        active_opponents = sum(1 for p in game.players if p.is_active) - 1
        win_rate = self.evaluator.calculate_win_probability(
            player.cards, game.board, 
            num_opponents=max(1, active_opponents), simulations=5
        )
        return win_rate

    def _quick_preflop_strength(self, cards: List[Card]) -> float:
        # 复用刚才的高性能版本代码
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
        # 复用之前的逻辑
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
        # 复用之前的逻辑
        action_idx = np.argmax(action_vector)
        
        # 移除强制弃牌逻辑，允许AI根据学习自行决定动作
        # 这使AI能够尝试用弱手牌到达摊牌阶段，有助于学习完整的扑克策略
        pass

        if action_idx == 0: return ('fold', None)
        elif action_idx == 1:
            call_amount = max(0, game.current_bet - player.bet)
            return ('call', call_amount)
        elif action_idx == 2:
            current_bet = max(game.current_bet, game.big_blind)
            amt = min(current_bet * 2, player.bet + player.stack)
            return ('raise', int(amt))
        elif action_idx == 3:
            current_bet = max(game.current_bet, game.big_blind)
            amt = min(current_bet * 4, player.bet + player.stack)
            return ('raise', int(amt))
        elif action_idx == 4:
            return ('raise', int(player.bet + player.stack))
        return ('fold', None)