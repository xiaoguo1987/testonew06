# 德州扑克自我对弈训练器（优化版）

import os
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
import pickle
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SelfPlayTrainer')

from ..game.poker_game import PokerGame, Player
from ..model.ai_agent import PokerAIAgent, EnsembleAgent
from ..reward.reward_calculator import RewardCalculator, AdaptiveRewardCalculator
from utils.checkpoint import save_checkpoint  # 确保该导入有效


class SelfPlayTrainer:
    """德州扑克自我对弈训练器（整合策略指标统计与自适应奖励调整）"""
    
    def __init__(self,
                 model_type: str = 'dqn',  # 'dqn'、'policy_gradient' 或 'ppo'
                 checkpoint_dir: str = './checkpoints',
                 small_blind: int = 5,
                 big_blind: int = 10,
                 initial_stack: int = 1000,
                 num_players: int = 6,
                 eval_interval: int = 100,  # 评估间隔（局数）
                 checkpoint_interval: int = 500,  # 保存检查点间隔（局数）
                 num_eval_games: int = 50,  # 每次评估的游戏局数
                 use_ensemble: bool = False,
                 opponent_update_interval: int = 10000,
                 learning_rate: float = 0.001,
                 optimizer: str = 'adam',
                 target_update_freq: int = 10):
        
        # 核心参数初始化
        self.model_type = model_type
        self.checkpoint_dir = checkpoint_dir
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.initial_stack = initial_stack
        self.num_players = num_players
        self.eval_interval = eval_interval
        self.checkpoint_interval = checkpoint_interval
        self.num_eval_games = num_eval_games
        self.use_ensemble = use_ensemble
        self.opponent_update_interval = opponent_update_interval
        self.target_update_freq = target_update_freq
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        
        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 创建AI代理
        self._create_agents()
        
        # 创建奖励计算器
        self.reward_calculator = AdaptiveRewardCalculator()
        
        # 统计信息（基础）
        self.total_hands = 0
        self.total_profit = 0
        self.best_score = -float('inf')
        self.performance_history = []
        
        # 新增：策略指标统计
        self.total_aggressive_actions = 0  # 进攻动作（raise/call）总数
        self.total_bluff_attempts = 0      # 诈唬尝试次数
        self.total_raise_actions = 0       # 加注动作总数
        self.total_invalid_bets = 0        # 无效下注次数
    
    def _create_agents(self) -> None:
        """创建AI代理（主代理+对手代理）"""
        if self.use_ensemble:
            self.main_agent = EnsembleAgent()
        else:
            self.main_agent = PokerAIAgent(
                agent_type=self.model_type, 
                target_update_freq=self.target_update_freq, 
                learning_rate=self.learning_rate, 
                optimizer=self.optimizer
            )
        
        # 对手代理（数量=玩家数-1）
        self.opponent_agents = []
        for i in range(self.num_players - 1):
            if self.use_ensemble:
                self.opponent_agents.append(EnsembleAgent())
            else:
                self.opponent_agents.append(PokerAIAgent(
                    agent_type=self.model_type, 
                    target_update_freq=self.target_update_freq, 
                    learning_rate=self.learning_rate, 
                    optimizer=self.optimizer
                ))
    
    def _create_game(self) -> Tuple[PokerGame, List[Player]]:
        """创建游戏实例和玩家（复用逻辑，避免重复初始化）"""
        players = [Player("Main_AI", self.initial_stack, True)]  # 主玩家
        players.extend([Player(f"Opponent_{i+1}", self.initial_stack, True) 
                       for i in range(self.num_players - 1)])  # 对手玩家
        return PokerGame(self.small_blind, self.big_blind, players), players
    
    def _count_episode_metrics(self, main_player: Player, game: PokerGame, 
                              action_history: List[Tuple[str, int]]) -> Dict[str, int]:
        """统计单局策略指标（整合到训练流程）"""
        bluffs = 0
        aggressive = 0
        raises = 0
        invalid_bets = 0

        for action, amount in action_history:
            # 统计进攻动作（raise/call）
            if action in ['raise', 'call']:
                aggressive += 1
            # 统计加注动作及衍生指标
            if action == 'raise':
                raises += 1
                # 判断诈唬（复用奖励计算器逻辑）
                if self.reward_calculator.is_bluffing(main_player, game, action):
                    bluffs += 1
                # 判断无效下注（基于手牌强度和底池赔率）
                hand_strength = self.reward_calculator.calculate_hand_strength(main_player, game)
                pot_odds = self.reward_calculator.calculate_pot_odds(main_player, game, amount)
                if hand_strength < 0.3 and pot_odds < 0.0 and amount > game.pot * 0.5:
                    invalid_bets += 1

        return {
            'bluffs': bluffs,
            'aggressive': aggressive,
            'raises': raises,
            'invalid_bets': invalid_bets
        }
    
    def _play_hand(self, game: PokerGame, players: List[Player], is_training: bool = True) -> Dict:
        """玩一手牌（修复索引风险，整合指标统计）"""
        action_histories = {p.name: [] for p in players}
        game.start_new_hand()  # 开始新的一手牌

        # 游戏主循环
        while not game.is_hand_over():  # 改用is_hand_over判断游戏结束（更准确）
            if game.is_round_over():
                if not game.proceed_to_next_round():
                    break  # 无法进入下一轮，结束本局
            current_player = game.next_player()
            if not current_player or not current_player.is_active:
                continue  # 跳过非活跃玩家

            # 选择代理
            if current_player.name == "Main_AI":
                agent = self.main_agent
            else:
                opp_idx = int(current_player.name.split('_')[-1]) - 1
                agent = self.opponent_agents[opp_idx] if opp_idx < len(self.opponent_agents) else self.opponent_agents[0]

            # 决策与执行动作
            action, amount = agent.select_action(game, current_player)
            action_histories[current_player.name].append((action, amount))
            game.handle_action(current_player, action, amount)

        # 计算结果
        winner = max(players, key=lambda p: p.stack)  # 简化：筹码最多者为赢家
        player_profits = {p.name: p.stack - self.initial_stack for p in players}
        result = {
            'winner': winner.name,
            'player_profits': player_profits,
            'action_histories': action_histories
        }

        # 训练模式：更新主代理与统计指标
        if is_training:
            main_player = players[0]
            main_profit = player_profits[main_player.name]
            is_main_winner = (winner.name == main_player.name)
            main_actions = action_histories[main_player.name]

            # 1. 统计单局策略指标并更新全局统计
            if main_actions:  # 避免无动作时的统计错误
                episode_metrics = self._count_episode_metrics(main_player, game, main_actions)
                self.total_bluff_attempts += episode_metrics['bluffs']
                self.total_aggressive_actions += episode_metrics['aggressive']
                self.total_raise_actions += episode_metrics['raises']
                self.total_invalid_bets += episode_metrics['invalid_bets']

            # 2. 计算奖励并更新代理（修复索引越界）
            for idx, (action, amount) in enumerate(main_actions):
                reward = self.reward_calculator.calculate_reward(
                    main_player, game, action, amount,
                    is_winner=is_main_winner, profit_loss=main_profit,
                    done=(idx == len(main_actions) - 1)  # 最后一个动作标记为done
                )
                self.main_agent.learn(reward, done=(idx == len(main_actions) - 1))

        return result
    
    def evaluate(self) -> Dict:
        """评估模型性能（新增策略指标输出）"""
        logger.info(f"开始评估，共{self.num_eval_games}局")
        total_profit = 0
        wins = 0
        eval_bluffs = 0
        eval_aggressive = 0

        for _ in range(self.num_eval_games):
            game, players = self._create_game()
            result = self._play_hand(game, players, is_training=False)
            total_profit += result['player_profits']['Main_AI']
            if result['winner'] == 'Main_AI':
                wins += 1
            # 统计评估中的策略指标
            main_actions = result['action_histories']['Main_AI']
            if main_actions:
                metrics = self._count_episode_metrics(players[0], game, main_actions)
                eval_bluffs += metrics['bluffs']
                eval_aggressive += metrics['aggressive']

        # 计算评估指标
        avg_profit = total_profit / self.num_eval_games
        win_rate = wins / self.num_eval_games
        bluff_rate = eval_bluffs / eval_aggressive if eval_aggressive > 0 else 0.0
        eval_result = {
            'avg_profit': avg_profit, 'win_rate': win_rate,
            'total_profit': total_profit, 'wins': wins,
            'bluff_rate': bluff_rate, 'total_hands': self.num_eval_games
        }

        # 日志输出评估结果
        logger.info(
            f"评估完成: 平均利润={avg_profit:.2f}, 胜率={win_rate:.2%}, "
            f"诈唬率={bluff_rate:.3f}"
        )
        self.performance_history.append(eval_result)
        return eval_result
    
    def update_opponents(self) -> None:
        """更新对手代理（优化文件操作安全性）"""
        if not self.opponent_agents:
            return
        opp_idx = random.randint(0, len(self.opponent_agents) - 1)
        temp_path = os.path.join(self.checkpoint_dir, f"temp_opponent_{opp_idx}.pkl")  # 通用格式
        
        try:
            self.main_agent.save_model(temp_path)  # 由代理自身决定保存格式
            self.opponent_agents[opp_idx].load_model(temp_path)
            logger.info(f"更新对手 {opp_idx + 1} 模型")
        except Exception as e:
            logger.error(f"更新对手失败: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """保存检查点（灵活处理文件名与格式）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if is_best:
            filename = f"best_agent_{self.model_type}_{timestamp}.pkl"
        else:
            filename = f"agent_{self.model_type}_{self.total_hands}.pkl"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        try:
            self.main_agent.save_model(filepath)
            logger.info(f"保存检查点: {filename}")
            # 保存训练状态
            state = {
                'total_hands': self.total_hands,
                'total_profit': self.total_profit,
                'best_score': self.best_score,
                'performance_history': self.performance_history,
                'strategy_metrics': {  # 新增策略指标状态
                    'bluff_attempts': self.total_bluff_attempts,
                    'aggressive_actions': self.total_aggressive_actions
                }
            }
            with open(filepath.replace('.pkl', '_state.pkl'), 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
    
    def load_checkpoint(self, filepath: Optional[str] = None) -> bool:
        """加载检查点（健壮解析文件名）"""
        if not filepath:
            # 优先加载最佳模型，再加载最新检查点
            best_models = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('best_agent_')]
            if best_models:
                filepath = os.path.join(self.checkpoint_dir, max(best_models))  # 最新最佳模型
            else:
                checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('agent_')]
                if checkpoints:
                    # 按文件名中的局数排序（提取数字部分）
                    def extract_hands(fname):
                        parts = fname.split('_')
                        for p in parts:
                            if p.replace('.pkl', '').isdigit():
                                return int(p.replace('.pkl', ''))
                        return 0
                    checkpoints.sort(key=extract_hands, reverse=True)
                    filepath = os.path.join(self.checkpoint_dir, checkpoints[0])
                else:
                    logger.warning("无检查点文件")
                    return False
        
        try:
            self.main_agent.load_model(filepath)
            logger.info(f"加载检查点: {filepath}")
            # 加载状态
            state_path = filepath.replace('.pkl', '_state.pkl')
            if os.path.exists(state_path):
                with open(state_path, 'rb') as f:
                    state = pickle.load(f)
                    self.total_hands = state.get('total_hands', 0)
                    self.total_profit = state.get('total_profit', 0)
                    self.best_score = state.get('best_score', -float('inf'))
                    self.performance_history = state.get('performance_history', [])
                    # 恢复策略指标
                    strat_metrics = state.get('strategy_metrics', {})
                    self.total_bluff_attempts = strat_metrics.get('bluff_attempts', 0)
                    self.total_aggressive_actions = strat_metrics.get('aggressive_actions', 0)
            return True
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return False
    
    def record_performance(self, profit_loss: float) -> None:
        """记录训练表现（计算策略指标比率，用于奖励调整）"""
        # 计算策略指标比率（避免除零错误）
        bluff_rate = self.total_bluff_attempts / self.total_aggressive_actions if self.total_aggressive_actions > 0 else 0.0
        invalid_bet_rate = self.total_invalid_bets / self.total_raise_actions if self.total_raise_actions > 0 else 0.0

        # 日志输出
        logger.info(
            f"训练统计 - 总手牌: {self.total_hands}, "
            f"诈唬率: {bluff_rate:.3f}, 无效下注率: {invalid_bet_rate:.3f}, "
            f"累计利润: {self.total_profit:.2f}"
        )

        # 传递指标给自适应奖励计算器
        if isinstance(self.reward_calculator, AdaptiveRewardCalculator):
            self.reward_calculator.record_performance(
                profit_loss=profit_loss,
                hands_played=1,
                bluff_rate=bluff_rate,
                invalid_bet_rate=invalid_bet_rate
            )
            # 每10局调整一次奖励权重
            if self.total_hands % 10 == 0:
                self.reward_calculator.adjust_rewards()
                logger.info(f"奖励权重更新: {self.reward_calculator.reward_weights}")
    
    def train(self, num_hands: int) -> None:
        """主训练循环（整合指标统计与评估）"""
        logger.info(f"开始训练，共{num_hands}手牌")
        self.load_checkpoint()  # 加载历史检查点
        start_hand = self.total_hands

        for _ in range(start_hand, start_hand + num_hands):
            game, players = self._create_game()
            result = self._play_hand(game, players, is_training=True)
            
            # 更新基础统计
            self.total_hands += 1
            self.total_profit += result['player_profits']['Main_AI']
            
            # 记录单局表现（含策略指标）
            self.record_performance(result['player_profits']['Main_AI'])
            
            # 定期评估
            if self.total_hands % self.eval_interval == 0:
                eval_result = self.evaluate()
                current_score = eval_result['avg_profit'] * eval_result['win_rate']
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.save_checkpoint(is_best=True)
                    logger.info(f"新最佳模型！得分: {current_score:.4f}")
            
            # 定期更新对手
            if self.total_hands % self.opponent_update_interval == 0:
                self.update_opponents()
            
            # 定期保存检查点
            if self.total_hands % self.checkpoint_interval == 0:
                self.save_checkpoint(is_best=False)
        
        logger.info(f"训练完成！总手牌: {self.total_hands}")
        self.save_checkpoint(is_best=False)  # 保存最终模型
        final_eval = self.evaluate()
        logger.info(f"最终评估: {final_eval}")


# 示例使用
if __name__ == "__main__":
    trainer = SelfPlayTrainer(
        model_type='dqn',
        checkpoint_dir='./checkpoints',
        small_blind=5,
        big_blind=10,
        initial_stack=1000,
        eval_interval=50,
        checkpoint_interval=200,
        num_eval_games=20
    )
    trainer.train(num_hands=100)