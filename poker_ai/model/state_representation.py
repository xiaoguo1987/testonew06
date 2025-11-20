# 德州扑克状态表示模块

import numpy as np
from typing import List, Dict, Optional
from ..game.cards import Card, RANK_VALUES
from ..game.poker_game import Player, PokerGame

class StateRepresentation:
    """德州扑克状态表示转换器"""
    
    def __init__(self):
        # 定义状态空间的各个部分的维度
        self.player_cards_dim = 13 * 4  # 玩家手牌的one-hot编码
        self.board_cards_dim = 13 * 4  # 公共牌的one-hot编码
        self.player_info_dim = 6 * 3  # 6个玩家的信息（筹码、下注、活跃状态）
        self.game_info_dim = 5  # 游戏信息（轮次、底池大小、当前下注、大盲注、小盲注）
        self.total_state_dim = (
            self.player_cards_dim + 
            self.board_cards_dim + 
            self.player_info_dim + 
            self.game_info_dim
        )
    
    def card_to_one_hot(self, cards: List[Card]) -> np.ndarray:
        """将牌转换为one-hot编码向量"""
        one_hot = np.zeros(13 * 4)
        for card in cards:
            # 计算牌在one-hot向量中的位置
            # 花色顺序: ♥, ♦, ♣, ♠ (0-3)
            suit_idx = ['♥', '♦', '♣', '♠'].index(card.suit)
            # 点数顺序: 2-A (0-12)
            rank_idx = list(RANK_VALUES.keys()).index(card.rank)
            position = rank_idx * 4 + suit_idx
            one_hot[position] = 1
        return one_hot
    
    def player_info_to_vector(self, players: List[Player], current_player_idx: int) -> np.ndarray:
        """将玩家信息转换为向量"""
        vector = np.zeros(self.player_info_dim)
        for i, player in enumerate(players):
            # 计算相对于当前玩家的位置（考虑到庄家位置的循环性）
            # 0: 当前玩家, 1-5: 其他玩家按行动顺序
            relative_pos = (i - current_player_idx) % 6
            base_idx = relative_pos * 3
            
            # 标准化筹码和下注金额（这里简单处理，实际应该基于大盲注标准化）
            # 0-1之间的值表示相对大小
            stack_normalized = min(player.stack / 10000, 1.0)  # 假设最大筹码为10000
            bet_normalized = min(player.bet / 1000, 1.0)  # 假设最大单轮下注为1000
            
            vector[base_idx] = stack_normalized
            vector[base_idx + 1] = bet_normalized
            vector[base_idx + 2] = 1.0 if player.is_active else 0.0
        return vector
    
    def game_info_to_vector(self, game: PokerGame) -> np.ndarray:
        """将游戏信息转换为向量"""
        vector = np.zeros(self.game_info_dim)
        
        # 轮次（翻牌前=0, 翻牌=1, 转牌=2, 河牌=3）
        vector[0] = (game.round - 1) / 4.0  # 归一化到0-1
        
        # 底池大小标准化
        vector[1] = min(game.pot / 10000, 1.0)
        
        # 当前下注标准化
        vector[2] = min(game.current_bet / 1000, 1.0)
        
        # 大盲注和小盲注
        vector[3] = game.big_blind / 100.0  # 假设最大大盲注为100
        vector[4] = game.small_blind / 100.0
        
        return vector
    
    def get_state(self, game: PokerGame, current_player: Player) -> np.ndarray:
        """获取完整的游戏状态向量"""
        # 获取当前玩家索引
        current_player_idx = game.players.index(current_player)
        
        # 1. 玩家手牌的one-hot编码
        player_cards_vector = self.card_to_one_hot(current_player.cards)
        
        # 2. 公共牌的one-hot编码
        board_cards_vector = self.card_to_one_hot(game.board)
        
        # 3. 玩家信息向量
        player_info_vector = self.player_info_to_vector(game.players, current_player_idx)
        
        # 4. 游戏信息向量
        game_info_vector = self.game_info_to_vector(game)
        
        # 组合所有向量
        state = np.concatenate([
            player_cards_vector,
            board_cards_vector,
            player_info_vector,
            game_info_vector
        ])
        
        return state
    
    def encode_action(self, action: str, amount: Optional[int] = None, game: Optional[PokerGame] = None) -> np.ndarray:
        """将动作编码为向量"""
        # 假设我们有5个离散动作:
        # 0: 弃牌, 1: 跟注, 2: 小加注(2倍), 3: 中加注(4倍), 4: 大加注(全下)
        action_vector = np.zeros(5)
        
        if action == 'fold':
            action_vector[0] = 1
        elif action == 'call':
            action_vector[1] = 1
        elif action == 'raise' and game is not None:
            if amount is None:
                raise ValueError("加注需要指定金额")
            
            current_bet = game.current_bet
            if current_bet == 0:
                # 翻牌前第一个行动，基于大盲注计算
                base_bet = game.big_blind
            else:
                base_bet = current_bet
            
            # 根据加注倍数决定使用哪个动作槽
            raise_ratio = amount / base_bet if base_bet > 0 else 0
            
            if raise_ratio <= 2.5:
                action_vector[2] = 1  # 小加注
            elif raise_ratio <= 5.0:
                action_vector[3] = 1  # 中加注
            else:
                action_vector[4] = 1  # 大加注/全下
        
        return action_vector
    
    def decode_action(self, action_vector: np.ndarray, game: PokerGame, player: Player) -> tuple:
        """将动作向量解码为具体动作"""
        # 找到概率最大的动作
        action_idx = np.argmax(action_vector)
        
        if action_idx == 0:
            return ('fold', None)
        elif action_idx == 1:
            # 跟注
            call_amount = max(0, game.current_bet - player.bet)
            return ('call', call_amount)
        elif action_idx == 2:
            # 小加注 (2倍当前下注)
            current_bet = game.current_bet
            if current_bet == 0:
                # 翻牌前第一个行动
                raise_amount = game.big_blind * 2
            else:
                raise_amount = current_bet * 2
            
            # 确保不超过玩家筹码
            raise_amount = min(raise_amount, player.bet + player.stack)
            return ('raise', raise_amount)
        elif action_idx == 3:
            # 中加注 (4倍当前下注)
            current_bet = game.current_bet
            if current_bet == 0:
                raise_amount = game.big_blind * 4
            else:
                raise_amount = current_bet * 4
            
            raise_amount = min(raise_amount, player.bet + player.stack)
            return ('raise', raise_amount)
        elif action_idx == 4:
            # 大加注/全下
            all_in_amount = player.bet + player.stack
            return ('raise', all_in_amount)
        else:
            # 默认弃牌
            return ('fold', None)

# 测试代码
if __name__ == "__main__":
    from ..game.poker_game import Player, PokerGame
    
    # 创建玩家
    players = [
        Player(f"Player {i}", 1000, i > 0) for i in range(6)
    ]
    
    # 创建游戏
    game = PokerGame(5, 10, players)
    game.start_new_hand()
    
    # 创建状态表示器
    state_rep = StateRepresentation()
    
    # 获取第一个玩家的状态
    state = state_rep.get_state(game, players[0])
    print(f"状态向量维度: {len(state)}")
    print(f"状态向量示例前10个元素: {state[:10]}")
    
    # 测试动作编码和解码
    action_vec = state_rep.encode_action('raise', 50, game)
    print(f"动作向量: {action_vec}")
    
    decoded_action = state_rep.decode_action(action_vec, game, players[0])
    print(f"解码动作: {decoded_action}")