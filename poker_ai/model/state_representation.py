# state_representation.py (6-Max 专用增强版)

import json
import os
import numpy as np
from typing import List, Dict, Optional
from ..game.cards import Card, RANK_VALUES
from ..game.poker_game import Player, PokerGame
from ..model.hand_evaluator import HandEvaluator
from ..action_space import (
    ACTION_DIM,
    ALLIN_ACTION_INDEX,
    POSTFLOP_POT_FRACTIONS,
    PREFLOP_RAISE_MULTIPLIERS,
    RAISE_ACTION_START,
)

# ---- 翻牌前胜率查找表 (LUT) 共享单例 ----
_PREFLOP_LUT: Optional[Dict] = None


def _get_preflop_lut() -> Dict:
    global _PREFLOP_LUT
    if _PREFLOP_LUT is not None:
        return _PREFLOP_LUT
    lut_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'preflop_equity.json')
    try:
        with open(lut_path, 'r') as f:
            _PREFLOP_LUT = json.load(f)
    except FileNotFoundError:
        _PREFLOP_LUT = {}
    return _PREFLOP_LUT

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

        # 7. [新增] 对手类型标识 (每座位1维: 0=fish, 1=ppo_opponent) = 6维
        # 解决多目标梯度冲突：让策略能区分不同对手类型
        self.opponent_type_dim = 6

        # 8. [新增] 对手行为统计 EMA (每座位4维: VPIP/PFR/AF/fold_to_3bet) = 24维
        # 让策略感知对手风格（松紧/激进被动），解决多对手适应问题
        self.opponent_behavior_dim = 24  # 6 seats × 4 stats

        self.total_state_dim = (
            self.player_cards_dim +
            self.board_cards_dim +
            self.player_info_dim +
            self.game_info_dim +
            self.position_one_hot_dim +
            self.active_count_dim +
            self.opponent_type_dim +
            self.opponent_behavior_dim
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

    def card_codes_to_one_hot(self, card_codes: np.ndarray) -> np.ndarray:
        """整数编码卡牌直接转 one-hot，避免高频 Card 对象解包。"""
        one_hot = np.zeros(52, dtype=np.float32)
        if len(card_codes) > 0:
            valid_codes = np.asarray(card_codes, dtype=np.int32)
            valid_codes = valid_codes[valid_codes >= 0]
            one_hot[valid_codes] = 1.0
        return one_hot
    
    def player_info_to_vector(self, players: List[Player], current_player_idx: int, game: PokerGame) -> np.ndarray:
        """玩家状态向量，保持相对位置顺序，让AI总是在第一个位置看到自己"""
        vector = np.zeros(self.player_info_dim)
        bb = max(game.big_blind, 1)
        stacks = getattr(game, 'player_stacks', None)
        bets = getattr(game, 'player_bets', None)
        active = getattr(game, 'player_active', None)
        
        for i in range(6): # 固定循环6次，确保维度固定
            # 这里的逻辑是：将当前玩家(Hero)放在向量的开头(i=0)，其他人顺时针排列
            # 这样 AI 永远知道 vector[0:3] 是自己的信息
            target_idx = (current_player_idx + i) % len(players)
            
            if target_idx < len(players):
                base_idx = i * 3
                
                # 归一化特征
                if stacks is not None and bets is not None and active is not None:
                    stack_normalized = min(float(stacks[target_idx]) / (bb * 100), 1.0)
                    bet_normalized = min(float(bets[target_idx]) / (bb * 100), 1.0)
                    is_active = 1.0 if bool(active[target_idx]) else 0.0
                else:
                    player = players[target_idx]
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

    def get_opponent_behavior_vector(self, game: PokerGame, current_player_idx: int,
                                      opponent_stats: np.ndarray = None) -> np.ndarray:
        """[新增] 获取对手行为统计特征向量 (6座位 × 4统计 = 24维)。

        每个座位编码 4 个 EMA 统计值：VPIP, PFR, AF, fold_to_3bet。
        旋转方向与 player_info_to_vector 一致：Hero 在位置 0。
        """
        vector = np.zeros(self.opponent_behavior_dim, dtype=np.float32)
        if opponent_stats is None or len(opponent_stats) == 0:
            return vector
        num_players = len(game.players)
        for i in range(min(6, num_players)):
            target_idx = (current_player_idx + i) % num_players
            if i == 0:
                continue  # Hero: all zeros
            if target_idx < len(opponent_stats):
                base = i * 4
                stats = opponent_stats[target_idx]
                vector[base:base+4] = np.clip(stats, 0.0, 1.0)
        return vector

    def get_active_count_vector(self, game: PokerGame) -> np.ndarray:
        """[新增] 获取活跃玩家数量特征"""
        if hasattr(game, 'player_active'):
            active_count = int(np.sum(game.player_active))
        else:
            active_count = sum(1 for p in game.players if p.is_active)
        # 归一化：(活跃人数 - 1) / 5.0。
        # 结果范围 0.0 (Heads-up) 到 1.0 (6人全在)
        normalized = (active_count - 1) / 5.0 if active_count > 0 else 0
        return np.array([normalized])

    def get_opponent_type_vector(self, game: PokerGame, current_player_idx: int,
                                  opponent_types: list = None) -> np.ndarray:
        """[新增] 获取每座位的对手类型特征向量。

        让策略能区分鱼玩家和 PPO 对手，解决多目标梯度冲突。
        - Position 0 (Hero): 固定 0.0
        - Positions 1-5: 0.0=fish, 1.0=ppo_opponent

        旋转方向与 player_info_to_vector 一致：Hero 在位置 0。
        """
        vector = np.zeros(self.opponent_type_dim, dtype=np.float32)
        if opponent_types is None:
            return vector
        num_players = len(game.players)
        for i in range(min(self.opponent_type_dim, num_players)):
            target_idx = (current_player_idx + i) % num_players
            if i == 0:
                # Hero: always 0
                vector[i] = 0.0
            else:
                # Map opponent to opp_type_list (5 entries for non-hero players)
                oi = target_idx - 1
                if oi >= 0 and oi < len(opponent_types):
                    vector[i] = 1.0 if opponent_types[oi] == 'ppo' else 0.0
        return vector

    def get_state(self, game: PokerGame, current_player: Player,
                  opponent_types: list = None,
                  opponent_stats: np.ndarray = None) -> np.ndarray:
        """获取完整的游戏状态向量"""
        current_player_idx = game.players.index(current_player)
        hole_codes = getattr(current_player, 'card_codes', None)
        board_codes = getattr(game, 'board_codes', None)
        
        # 组合所有特征
        state = np.concatenate([
            # 1. 自身手牌
            self.card_codes_to_one_hot(hole_codes) if hole_codes is not None else self.card_to_one_hot(current_player.cards),
            # 2. 公共牌
            self.card_codes_to_one_hot(board_codes) if board_codes is not None else self.card_to_one_hot(game.board),
            # 3. 所有玩家状态 (Hero排在最前)
            self.player_info_to_vector(game.players, current_player_idx, game),
            # 4. 游戏全局信息
            self.game_info_to_vector(game),
            # 5. [新增] 绝对位置信息 (至关重要)
            self.get_position_vector(game, current_player),
            # 6. [新增] 活跃人数信息
            self.get_active_count_vector(game),
            # 7. [新增] 对手类型标识 (区分鱼/PPO对手)
            self.get_opponent_type_vector(game, current_player_idx, opponent_types),
            # 8. [新增] 对手行为统计 (VPIP/PFR/AF/fold_to_3bet)
            self.get_opponent_behavior_vector(game, current_player_idx, opponent_stats),
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
        """翻牌前手牌强度：优先查LUT（精确MC胜率），fallback到评分公式"""
        if len(cards) != 2: return 0.0
        c1, c2 = sorted(cards, key=lambda x: x.value, reverse=True)
        is_suited = c1.suit == c2.suit
        if c1.value == c2.value:
            key = f"{c1.rank}{c2.rank}"
        else:
            key = f"{c1.rank}{c2.rank}{'s' if is_suited else 'o'}"
        if key in self._preflop_cache: return self._preflop_cache[key]

        # 优先查LUT
        lut = _get_preflop_lut()
        if lut and key in lut:
            # 默认取 vs 3 opponents（6人桌中位情况）
            strength = float(lut[key].get('3', lut[key].get('1', 0.5)))
            self._preflop_cache[key] = strength
            return strength

        # Fallback: 评分公式
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
        action_vector = np.zeros(ACTION_DIM)
        if action == 'fold':
            action_vector[0] = 1
        elif action == 'call':
            action_vector[1] = 1
        elif action == 'raise' and game is not None:
            if amount is not None and game is not None:
                stacks = getattr(game, 'player_stacks', None)
                try:
                    current_player = game.players[game.current_player_idx]
                    player_idx = game.players.index(current_player)
                    player_bet = (
                        int(game.player_bets[player_idx])
                        if hasattr(game, 'player_bets')
                        else current_player.bet
                    )
                    player_stack = (
                        int(stacks[player_idx])
                        if stacks is not None
                        else current_player.stack
                    )
                    if amount >= player_bet + player_stack:
                        action_vector[ALLIN_ACTION_INDEX] = 1
                        return action_vector
                except Exception:
                    pass

            if game.round == 1:
                # 翻牌前: 按 base 倍数分类
                base = max(game.current_bet, game.big_blind)
                if amount is None:
                    action_vector[RAISE_ACTION_START] = 1
                    return action_vector
                ratio = amount / base if base > 0 else 0
                idx = int(np.argmin(np.abs(np.asarray(PREFLOP_RAISE_MULTIPLIERS) - ratio)))
                action_vector[RAISE_ACTION_START + idx] = 1
            else:
                # 翻牌后: 按 pot 比例分类
                cb = max(game.current_bet, game.big_blind)
                pot = max(game.pot, game.big_blind)
                if amount is None:
                    action_vector[RAISE_ACTION_START] = 1
                    return action_vector
                ratio = max(0.0, (amount - cb) / pot)
                idx = int(np.argmin(np.abs(np.asarray(POSTFLOP_POT_FRACTIONS) - ratio)))
                action_vector[RAISE_ACTION_START + idx] = 1
        return action_vector

    def decode_action(self, action_vector: np.ndarray, game: PokerGame, player: Player, hand_strength: float = 0.0) -> tuple:
        action_idx = np.argmax(action_vector)

        if action_idx == 0:
            return ('fold', None)
        elif action_idx == 1:
            call_amount = max(0, game.current_bet - player.bet)
            return ('call', call_amount)
        elif RAISE_ACTION_START <= action_idx < ALLIN_ACTION_INDEX:
            cb = max(game.current_bet, game.big_blind)
            bucket_idx = action_idx - RAISE_ACTION_START
            if game.round == 1:
                amt = int(cb * PREFLOP_RAISE_MULTIPLIERS[bucket_idx])
            else:
                pot = max(game.pot, game.big_blind)
                amt = int(cb + pot * POSTFLOP_POT_FRACTIONS[bucket_idx])
            amt = min(amt, player.bet + player.stack)
            min_raise = max(game.current_bet * 2, game.big_blind)
            if amt <= game.current_bet or amt < min_raise:
                return ('call', max(0, game.current_bet - player.bet))
            return ('raise', int(amt))
        elif action_idx == ALLIN_ACTION_INDEX:
            amt = player.bet + player.stack
            if amt < game.current_bet:
                return ('call', max(0, game.current_bet - player.bet))
            return ('raise', int(amt))
        return ('fold', None)
