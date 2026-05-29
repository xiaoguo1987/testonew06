# 德州扑克游戏核心逻辑

import random
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from .cards import Card, Deck, compare_hands, card_to_code, code_to_card

class Player:
    """德州扑克玩家类"""
    def __init__(self, name: str, stack: int, is_ai: bool = False):
        self.name = name
        self.is_ai = is_ai  # 是否是AI玩家
        self.total_profit = 0  # 累计利润，用于测试和分析
        self._stack = int(stack)
        self._bet = 0
        self._is_active = True
        self._is_all_in = False
        self._position = -1
        self._table_state = None
        self._seat_idx = -1
        self._hole_card_codes = np.full(2, -1, dtype=np.int16)
        self._hole_card_count = 0
        self._cards_cache: List[Card] = []
        self._cards_dirty = False

    def _attach_table_state(self, table_state: Dict[str, np.ndarray], seat_idx: int) -> None:
        self._table_state = table_state
        self._seat_idx = int(seat_idx)

    @property
    def stack(self) -> int:
        if self._table_state is None:
            return int(self._stack)
        return int(self._table_state['stacks'][self._seat_idx])

    @stack.setter
    def stack(self, value: int) -> None:
        value = int(value)
        if self._table_state is None:
            self._stack = value
        else:
            self._table_state['stacks'][self._seat_idx] = value

    @property
    def bet(self) -> int:
        if self._table_state is None:
            return int(self._bet)
        return int(self._table_state['bets'][self._seat_idx])

    @bet.setter
    def bet(self, value: int) -> None:
        value = int(value)
        if self._table_state is None:
            self._bet = value
        else:
            self._table_state['bets'][self._seat_idx] = value

    @property
    def is_active(self) -> bool:
        if self._table_state is None:
            return bool(self._is_active)
        return bool(self._table_state['active'][self._seat_idx])

    @is_active.setter
    def is_active(self, value: bool) -> None:
        value = bool(value)
        if self._table_state is None:
            self._is_active = value
        else:
            self._table_state['active'][self._seat_idx] = value

    @property
    def is_all_in(self) -> bool:
        if self._table_state is None:
            return bool(self._is_all_in)
        return bool(self._table_state['all_in'][self._seat_idx])

    @is_all_in.setter
    def is_all_in(self, value: bool) -> None:
        value = bool(value)
        if self._table_state is None:
            self._is_all_in = value
        else:
            self._table_state['all_in'][self._seat_idx] = value

    @property
    def position(self) -> int:
        if self._table_state is None:
            return int(self._position)
        return int(self._table_state['positions'][self._seat_idx])

    @position.setter
    def position(self, value: int) -> None:
        value = int(value)
        if self._table_state is None:
            self._position = value
        else:
            self._table_state['positions'][self._seat_idx] = value

    @property
    def card_codes(self) -> np.ndarray:
        return self._hole_card_codes[:self._hole_card_count].copy()

    @property
    def cards(self) -> List[Card]:
        if self._cards_dirty:
            self._cards_cache = [
                code_to_card(code)
                for code in self._hole_card_codes[:self._hole_card_count]
                if code >= 0
            ]
            self._cards_dirty = False
        return self._cards_cache

    @cards.setter
    def cards(self, value: List[Card]) -> None:
        self.clear_cards()
        for idx, card in enumerate(value[:2]):
            self.set_card(idx, card)

    def clear_cards(self) -> None:
        self._hole_card_codes.fill(-1)
        self._hole_card_count = 0
        self._cards_cache = []
        self._cards_dirty = False

    def set_card_code(self, idx: int, card_code: int) -> None:
        self._hole_card_codes[int(idx)] = int(card_code)
        self._hole_card_count = max(self._hole_card_count, int(idx) + 1)
        self._cards_dirty = True

    def set_card(self, idx: int, card: Card) -> None:
        self.set_card_code(idx, card_to_code(card))
    
    def add_bet(self, amount: int) -> None:
        """添加下注金额"""
        if amount > self.stack:
            raise ValueError(f"下注金额({amount})超过玩家筹码({self.stack})")
        self.bet += amount
        self.stack -= amount
        if self.stack == 0:
            self.is_all_in = True
    
    def reset_for_new_round(self) -> None:
        """为新一轮重置状态"""
        self.clear_cards()
        self.bet = 0
        self.is_active = True
        self.is_all_in = False
    
    def __str__(self) -> str:
        return f"{self.name} (筹码: {self.stack}, 位置: {self.position})"

    @property
    def is_human(self) -> bool:
        """向后兼容旧 demo/advisor 代码：非 AI 玩家视为 human。"""
        return not self.is_ai

    @is_human.setter
    def is_human(self, value: bool) -> None:
        self.is_ai = not bool(value)

class PokerGame:
    """德州扑克游戏管理器"""
    def __init__(self, small_blind: int, big_blind: int, players: List[Player]):
        if len(players) != 6:
            raise ValueError("这是一个6人桌德州扑克游戏")
        
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.players = players
        self.deck = Deck()
        self._board_codes = np.full(5, -1, dtype=np.int16)
        self._board_count = 0
        self._board_cache: List[Card] = []
        self._board_dirty = False
        self.pot = 0  # 底池
        self.current_bet = 0  # 当前轮次最大下注
        self.dealer_pos = 0  # 庄家位置
        self.current_player = 0  # 当前行动玩家
        self.round = 0  # 0: 预处理, 1: 翻牌前, 2: 翻牌, 3: 转牌, 4: 河牌
        self.action_count = 0  # 当前轮次行动次数
        self.winner = None
        self.winners: List[Player] = []
        self.player_stacks = np.array([int(player.stack) for player in players], dtype=np.int32)
        self.player_bets = np.zeros(len(players), dtype=np.int32)
        self.player_active = np.ones(len(players), dtype=np.bool_)
        self.player_all_in = np.zeros(len(players), dtype=np.bool_)
        self.player_positions = np.arange(len(players), dtype=np.int8)
        self._table_state = {
            'stacks': self.player_stacks,
            'bets': self.player_bets,
            'active': self.player_active,
            'all_in': self.player_all_in,
            'positions': self.player_positions,
        }
        
        # 设置玩家位置
        for i, player in enumerate(players):
            player._attach_table_state(self._table_state, i)
            player.position = i

    @property
    def board_codes(self) -> np.ndarray:
        return self._board_codes[:self._board_count].copy()

    @property
    def board(self) -> List[Card]:
        if self._board_dirty:
            self._board_cache = [
                code_to_card(code)
                for code in self._board_codes[:self._board_count]
                if code >= 0
            ]
            self._board_dirty = False
        return self._board_cache

    @board.setter
    def board(self, value: List[Card]) -> None:
        self._clear_board()
        for card in value[:5]:
            self._append_board_card(card_to_code(card))

    @property
    def community_cards(self) -> List[Card]:
        """公共牌别名，兼容旧接口。"""
        return self.board

    @community_cards.setter
    def community_cards(self, value: List[Card]) -> None:
        self.board = value

    @property
    def round_num(self) -> int:
        """旧接口使用 0-3 表示翻牌前/翻牌/转牌/河牌。"""
        return max(0, int(self.round) - 1)

    def _clear_board(self) -> None:
        self._board_codes.fill(-1)
        self._board_count = 0
        self._board_cache = []
        self._board_dirty = False

    def _append_board_card(self, card_code: int) -> None:
        self._board_codes[self._board_count] = int(card_code)
        self._board_count += 1
        self._board_dirty = True
    
    def start_new_hand(self) -> None:
        """开始新的一手牌"""
        # 重置游戏状态
        self.deck = Deck()
        self._clear_board()
        self.pot = 0
        self.current_bet = 0
        self.action_count = 0
        self.winner = None
        self.winners = []
        
        # 重置玩家状态
        self.player_bets.fill(0)
        self.player_active.fill(True)
        self.player_all_in.fill(False)
        for player in self.players:
            player.clear_cards()
        
        # 庄家位置顺时针移动
        self.dealer_pos = (self.dealer_pos + 1) % len(self.players)
        
        # 处理大小盲注
        self._post_blinds()
        
        # 发牌给玩家
        self._deal_hole_cards()
        
        # 开始翻牌前
        self.round = 1
        self._setup_round_start()
    
    def _post_blinds(self) -> None:
        """处理大小盲注"""
        # 小盲位置
        sb_pos = (self.dealer_pos + 1) % len(self.players)
        # 大盲位置
        bb_pos = (self.dealer_pos + 2) % len(self.players)
        
        # 下小盲注
        if self.players[sb_pos].stack >= self.small_blind:
            self.players[sb_pos].add_bet(self.small_blind)
            self.pot += self.small_blind
            self.current_bet = self.small_blind
        
        # 下大盲注
        if self.players[bb_pos].stack >= self.big_blind:
            self.players[bb_pos].add_bet(self.big_blind)
            self.pot += self.big_blind
            self.current_bet = self.big_blind
    
    def _deal_hole_cards(self) -> None:
        """发给每个玩家两张手牌"""
        for card_idx in range(2):
            for player in self.players:
                player.set_card_code(card_idx, self.deck.deal_code())
    
    def _setup_round_start(self) -> None:
        """设置轮次开始时的状态"""
        # 翻牌前必须保留盲注形成的当前下注；翻牌后新街才重置为0
        if self.round != 1:
            self.current_bet = 0
        self.action_count = 0
        
        # 确定起始行动玩家
        if self.round == 1:  # 翻牌前
            # 从大盲注后的玩家开始
            # next_player() 会先 +1 再返回，因此这里存上一个位置
            self.current_player = (self.dealer_pos + 2) % len(self.players)
        else:
            # 翻牌后从庄家位置的下一位开始
            self.current_player = self.dealer_pos
    
    def handle_action(self, player: Player, action: str, amount: Optional[int] = None) -> bool:
        """
        处理玩家行动
        action: 'fold', 'call', 'raise'
        返回: 是否成功处理行动
        """
        if not player.is_active or player.is_all_in:
            return False
        
        if action == 'fold':
            player.is_active = False
            self.action_count += 1
            return True
        
        elif action == 'call':
            call_amount = max(0, self.current_bet - player.bet)
            if call_amount > player.stack:
                # 只能全下
                actual_bet = player.stack
                player.add_bet(actual_bet)
                self.pot += actual_bet
            else:
                player.add_bet(call_amount)
                self.pot += call_amount
            
            self.action_count += 1
            return True
        
        elif action == 'raise':
            if amount is None or amount <= self.current_bet:
                return False
            
            # 确保下注合法（至少是大盲注的大小，除非全下）
            min_raise = max(self.current_bet * 2, self.big_blind)
            if amount < min_raise and amount < player.stack:
                return False
            
            # 确保玩家有足够的筹码
            raise_amount = amount - player.bet
            if raise_amount > player.stack:
                return False
            
            player.add_bet(raise_amount)
            self.pot += raise_amount
            self.current_bet = amount
            
            # 重置行动计数，因为有人加注
            self.action_count = 1
            return True
        
        return False
    
    def next_player(self) -> Optional[Player]:
        """移动到下一个活跃玩家"""
        for _ in range(len(self.players)):
            self.current_player = (self.current_player + 1) % len(self.players)
            player = self.players[self.current_player]
            if player.is_active and not player.is_all_in:
                return player
        return None  # 没有活跃玩家了
    
    def is_round_over(self) -> bool:
        """检查当前轮次是否结束"""
        # 计算活跃且未全下的玩家数量
        active_players = [p for p in self.players if p.is_active and not p.is_all_in]
        
        if len(active_players) == 0:
            return True
        
        # 检查所有活跃玩家是否都已下注相同金额
        if self.action_count >= len(active_players):
            # 检查所有活跃且未全下的玩家是否都已跟注
            all_called = all(p.bet == self.current_bet for p in active_players)
            return all_called
        
        return False
    
    def proceed_to_next_round(self) -> bool:
        """进入下一个游戏阶段"""
        # 如果只剩一个玩家，直接结束游戏
        active_players = [p for p in self.players if p.is_active]
        if len(active_players) == 1:
            self._award_pot(active_players[0])
            return True
        
        if self.round == 1:  # 翻牌前 -> 翻牌
            # 发三张公共牌
            for _ in range(3):
                self._append_board_card(self.deck.deal_code())
            self.round = 2
            self._setup_round_start()
        elif self.round == 2:  # 翻牌 -> 转牌
            self._append_board_card(self.deck.deal_code())
            self.round = 3
            self._setup_round_start()
        elif self.round == 3:  # 转牌 -> 河牌
            self._append_board_card(self.deck.deal_code())
            self.round = 4
            self._setup_round_start()
        elif self.round == 4:  # 河牌 -> 摊牌
            self._showdown()
            return True
        
        return False
    
    def _showdown(self) -> None:
        """摊牌并确定获胜者（基于 7 张牌最佳 5 张比较）"""
        active_players = [p for p in self.players if p.is_active]
        
        if not active_players:
            return

        # 使用 compare_hands（内部会从 7 张牌中选最佳 5 张）逐个比较，正确处理平局
        winners = [active_players[0]]
        for challenger in active_players[1:]:
            challenger_cards = challenger.cards + self.board
            best_cards = winners[0].cards + self.board
            cmp = compare_hands(challenger_cards, best_cards)
            if cmp > 0:
                winners = [challenger]
            elif cmp == 0:
                winners.append(challenger)
        
        # 3. 分配底池
        if len(winners) > 0:
            share = self.pot // len(winners)
            remainder = self.pot % len(winners)
            
            for winner in winners:
                winner.stack += share
                # 记录该玩家本局赢利，用于后续奖励计算
                # 注意：这里可能需要额外的逻辑标记谁是Winner，原代码是通过stack变化计算的
            
            # 剩下的零头给第一个人（简化处理）
            if remainder > 0:
                winners[0].stack += remainder
                
            self.pot = 0
            # 标记赢家供外部读取
            self.winner = winners[0] # 兼容旧逻辑：保留单一winner
            self.winners = winners
            
    def _award_pot(self, winner: Player) -> None:
        """奖励底池给获胜者"""
        winner.stack += self.pot
        self.pot = 0
        self.winner = winner
        self.winners = [winner]
    
    def is_hand_over(self) -> bool:
        """检查当前手牌是否结束"""
        # 检查是否只剩一个活跃玩家
        active_players = [p for p in self.players if p.is_active]
        if len(active_players) <= 1:
            return True
        
        # 检查是否已经完成所有轮次（河牌后）
        return self.round > 4
    
    def get_game_state(self) -> Dict:
        """获取游戏当前状态（用于AI决策）"""
        return {
            'round': self.round,
            'pot': self.pot,
            'current_bet': self.current_bet,
            'board': self.board,
            'players': [{
                'name': p.name,
                'stack': p.stack,
                'bet': p.bet,
                'is_active': p.is_active,
                'is_all_in': p.is_all_in,
                'position': p.position
            } for p in self.players]
        }

# 示例使用
if __name__ == "__main__":
    # 创建6个玩家
    players = [
        Player("Player 1", 1000, False),
        Player("Player 2", 1000, True),
        Player("Player 3", 1000, True),
        Player("Player 4", 1000, True),
        Player("Player 5", 1000, True),
        Player("Player 6", 1000, True)
    ]
    
    # 创建游戏
    game = PokerGame(small_blind=5, big_blind=10, players=players)
    
    # 开始游戏
    game.start_new_hand()
    
    # 显示游戏状态
    print(f"庄家位置: {game.dealer_pos}")
    print(f"翻牌前 - 公共牌: {game.board}")
    for player in game.players:
        print(f"{player.name}: 手牌 {player.cards}, 筹码 {player.stack}, 下注 {player.bet}")
