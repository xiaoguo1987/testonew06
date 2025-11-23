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
from ..utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from ..reward.simple_reward_calculator import SimpleRewardCalculator

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
                 target_update_freq: int = 100):
        
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
        # 修改前
        # self.reward_calculator = AdaptiveRewardCalculator()

        # 修改后
        self.reward_calculator = SimpleRewardCalculator()
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
        """玩一手牌（修复版：增加动作记录 + 优化无效动作处理）"""
        
        game.start_new_hand()
        last_step_info = None 
        initial_stack = players[0].stack 
        
        # --- 修复 1: 初始化动作记录字典 ---
        action_records = {p.name: [] for p in players}

        while not game.is_hand_over():
            if game.is_round_over():
                if not game.proceed_to_next_round():
                    break
            
            current_player = game.next_player()
            if not current_player or not current_player.is_active:
                continue

            # === 主AI 行动逻辑 ===
            if current_player.name == "Main_AI":
                current_state = self.main_agent.state_rep.get_state(game, current_player)
                
                if is_training and last_step_info is not None:
                    prev_state, prev_action_idx = last_step_info
                    # 给一个微小的生存奖励，鼓励它多玩几轮，不要无脑弃牌
                    self.main_agent.learn(0.01, prev_state, prev_action_idx, current_state, False)

                action_str, amount = self.main_agent.select_action(game, current_player)
                
                # 获取 action index
                decision_info = self.main_agent.get_decision_info(game, current_player)
                action_idx = decision_info.get('best_action_index', 0)
                if 'most_likely_action' in decision_info:
                    action_idx = decision_info['most_likely_action']

                # --- 修复 2: 在执行前记录动作 ---
                # 这样即使动作失败，我们也能在日志里看到它尝试过了
                action_records[current_player.name].append((action_str, amount if amount else 0))

                # 执行动作
                success = game.handle_action(current_player, action_str, amount)
                
                # 死循环保护
                if not success:
                    # 如果动作无效，强制弃牌，但不要给太大惩罚，否则它会不敢动
                    game.handle_action(current_player, 'fold', None)
                    if is_training and last_step_info is not None:
                        self.main_agent.learn(
                            reward=-0.1,  # 轻微惩罚即可
                            state=prev_state,
                            action=prev_action_idx,
                            next_state=current_state,
                            done=False
                        )
                
                last_step_info = (current_state, action_idx)

            # === 对手行动逻辑 ===
            else:
                opp_idx = int(current_player.name.split('_')[-1]) - 1
                agent = self.opponent_agents[opp_idx] if opp_idx < len(self.opponent_agents) else self.opponent_agents[0]
                action, amount = agent.select_action(game, current_player)
                # 对手也要记录动作，不然日志会缺
                action_records[current_player.name].append((action, amount if amount else 0))
                
                if not game.handle_action(current_player, action, amount):
                    game.handle_action(current_player, 'fold', None)

        # === 结算 ===
        final_profit = players[0].stack - initial_stack
        
        result = {
            'winner': game.winner.name if hasattr(game, 'winner') else 'Unknown',
            'player_profits': {p.name: p.stack - self.initial_stack for p in players},
            'action_histories': action_records  # --- 修复 3: 返回填充好的记录 ---
        }

        if is_training and last_step_info is not None:
            prev_state, prev_action_idx = last_step_info
            final_reward = final_profit / self.big_blind 
            terminal_state = np.zeros_like(prev_state)
            self.main_agent.learn(final_reward, prev_state, prev_action_idx, terminal_state, True)

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
        """更新对手代理（内存复制版：极速且无递归风险）"""
        if not self.opponent_agents:
            return
            
        try:
            # 随机选择一个对手进行更新
            opp_idx = random.randint(0, len(self.opponent_agents) - 1)
            
            # 1. 从主代理获取权重 (内存操作)
            weights = self.main_agent.get_weights()
            
            if weights is not None:
                # 2. 复制给对手代理 (内存操作)
                self.opponent_agents[opp_idx].set_weights(weights)
                logger.info(f"更新对手 {opp_idx + 1} 模型 (内存复制)")
            else:
                logger.warning("主代理不支持权重获取，跳过对手更新")
                
        except Exception as e:
            logger.error(f"更新对手失败: {e}")

    def update_opponentsold(self) -> None:
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
        """保存检查点（使用导入的工具函数）"""
        try:
            # 准备检查点文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if is_best:
                filename = f"best_agent_{self.model_type}_{timestamp}.pkl"
            else:
                filename = f"agent_{self.model_type}_{self.total_hands}.pkl"
            filepath = os.path.join(self.checkpoint_dir, filename)
            
            # 准备元数据
            metadata = {
                'total_hands': self.total_hands,
                'total_profit': self.total_profit,
                'best_score': self.best_score,
                'performance_history': self.performance_history,
                'strategy_metrics': {
                    'bluff_attempts': self.total_bluff_attempts,
                    'aggressive_actions': self.total_aggressive_actions
                },
                'model_type': self.model_type
            }
            
            # 使用导入的工具函数保存检查点
            success = save_checkpoint(self.main_agent, filepath, metadata)
            
            if success:
                logger.info(f"保存检查点: {filename}")
            else:
                # 后备方案：如果工具函数失败，使用原始方法
                logger.warning("工具函数保存失败，使用后备方法")
                self.main_agent.save_model(filepath)
                with open(filepath.replace('.pkl', '_state.pkl'), 'wb') as f:
                    pickle.dump(metadata, f)
                logger.info(f"使用后备方法保存检查点: {filename}")
                
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
    
    def load_checkpoint(self, filepath: Optional[str] = None) -> bool:
        """加载检查点（不自动加载最新检查点）"""
        # 不自动加载最新检查点，只在指定路径时加载
        if filepath is None:
            logger.warning("未指定检查点路径，不自动加载最新检查点")
            return False
        
        try:
            # 验证文件是否存在
            if not os.path.exists(filepath):
                # 尝试添加常见扩展名
                if os.path.exists(filepath + '.pkl'):
                    filepath = filepath + '.pkl'
                elif os.path.exists(filepath + '.weights'):
                    filepath = filepath + '.weights'
                else:
                    logger.error(f"检查点文件不存在: {filepath}")
                    return False
                    
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
        # 不再自动加载检查点，避免覆盖用户指定的模型
        logger.info("从当前模型状态开始训练")
        start_hand = self.total_hands

        for _ in range(start_hand, start_hand + num_hands):
            game, players = self._create_game()
            result = self._play_hand(game, players, is_training=True)
            
            # 更新基础统计
            self.total_hands += 1
            self.total_profit += result['player_profits']['Main_AI']
            
            # 记录单局表现（含策略指标）
            if self.total_hands % 100 == 0:
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