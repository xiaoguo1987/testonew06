# 德州扑克自我对弈训练器（优化版 - 最终修正）

import os
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
import pickle
import logging
import traceback
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
    """德州扑克自我对弈训练器（6-Max 极速训练版）"""
    
    def __init__(self,
                 model_type: str = 'dqn',
                 checkpoint_dir: str = './checkpoints',
                 small_blind: int = 5,
                 big_blind: int = 10,
                 initial_stack: int = 1000,
                 num_players: int = 6,  # 默认必须是6
                 eval_interval: int = 200,
                 checkpoint_interval: int = 1000,
                 num_eval_games: int = 100,
                 use_ensemble: bool = False,
                 opponent_update_interval: int = 5000,
                 learning_rate: float = 0.001,
                 optimizer: str = 'adam',
                 target_update_freq: int = 100):
        
        # --- 修正1: 强制检查 6-Max ---
        if num_players != 6:
            raise ValueError(f"当前 StateRepresentation 是 6-Max 专用的，必须设置 num_players=6，但你设置了 {num_players}")

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
        
        # 使用更复杂的奖励计算器，提供中间步骤奖励
        # 这有助于AI学习更精细的策略，而不仅是最终结果
        self.reward_calculator = RewardCalculator()

        # 统计信息（基础）
        self.total_hands = 0
        self.total_profit = 0
        self.best_score = -float('inf')
        self.performance_history = []
        
        # 策略指标统计
        self.total_aggressive_actions = 0
        self.total_bluff_attempts = 0
        self.total_raise_actions = 0
        self.total_invalid_bets = 0
    
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
        
        self.opponent_agents = []
        
        # 前 4 个对手是正常的 AI (历史版本)
        # 注意：这里 range 是 num_players - 2，因为除了主AI，还有一个Fish，剩下4个位置
        for i in range(self.num_players - 2):
            self.opponent_agents.append(PokerAIAgent(
                agent_type=self.model_type,
                target_update_freq=self.target_update_freq,
                learning_rate=self.learning_rate,
                optimizer=self.optimizer
            ))
            
        # 第 5 个对手作为“鱼” (Random Player)
        fish_agent = PokerAIAgent(agent_type=self.model_type)
        # 强制设置 epsilon 为 1.0 (全随机)
        if hasattr(fish_agent.agent, 'epsilon'):
            fish_agent.agent.epsilon = 1.0 
            fish_agent.agent.epsilon_min = 1.0
            fish_agent.agent.epsilon_decay = 0.0 # 不衰减
        
        self.opponent_agents.append(fish_agent)
    
    def _create_game(self) -> Tuple[PokerGame, List[Player]]:
        """创建游戏实例和玩家"""
        players = [Player("Main_AI", self.initial_stack, True)]
        # 创建对手玩家 (注意名字要和 opponent_agents 的索引对应)
        # Opponent_1 对应 agent[0], ..., Opponent_5 对应 agent[4] (Fish)
        players.extend([Player(f"Opponent_{i+1}", self.initial_stack, True) 
                       for i in range(self.num_players - 1)])
        return PokerGame(self.small_blind, self.big_blind, players), players
    
    def _count_episode_metrics(self, main_player: Player, game: PokerGame, 
                              action_history: List[Tuple[str, int]]) -> Dict[str, int]:
        """统计单局策略指标"""
        bluffs = 0
        aggressive = 0
        raises = 0
        invalid_bets = 0

        # --- 修正2: 只有当 reward_calculator 支持复杂逻辑时才计算 ---
        # SimpleRewardCalculator 是极简版，可能没有 is_bluffing 等方法
        # 这里做一个简单的兼容性检查
        has_complex_logic = hasattr(self.reward_calculator, 'is_bluffing')

        for action, amount in action_history:
            if action in ['raise', 'call']:
                aggressive += 1
            if action == 'raise':
                raises += 1
                if has_complex_logic:
                    if self.reward_calculator.is_bluffing(main_player, game, action):
                        bluffs += 1
                    # 判断无效下注
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
        """玩一手牌"""
        game.start_new_hand()
        last_step_info = None 
        initial_stack = players[0].stack 
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
                
                # 生存奖励 (微小)
                if is_training and last_step_info is not None:
                    prev_state, prev_action_idx = last_step_info
                    self.main_agent.learn(0.001, prev_state, prev_action_idx, current_state, False)

                action_str, amount = self.main_agent.select_action(game, current_player)
                
                decision_info = self.main_agent.get_decision_info(game, current_player)
                action_idx = decision_info.get('best_action_index', 0)
                if 'most_likely_action' in decision_info:
                    action_idx = decision_info['most_likely_action']

                action_records[current_player.name].append((action_str, amount if amount else 0))

                success = game.handle_action(current_player, action_str, amount)
                
                if not success:
                    game.handle_action(current_player, 'fold', None)
                    # 动作无效惩罚
                    if is_training and last_step_info is not None:
                        self.main_agent.learn(
                            reward=-0.05, 
                            state=prev_state,
                            action=prev_action_idx,
                            next_state=current_state,
                            done=False
                        )
                
                last_step_info = (current_state, action_idx)

            # === 对手行动逻辑 ===
            else:
                # 解析名字获取索引 "Opponent_X" -> index X-1
                try:
                    opp_idx = int(current_player.name.split('_')[-1]) - 1
                    # 确保索引不越界
                    if 0 <= opp_idx < len(self.opponent_agents):
                        agent = self.opponent_agents[opp_idx]
                    else:
                        agent = self.opponent_agents[0] # Fallback
                except:
                    agent = self.opponent_agents[0]

                action, amount = agent.select_action(game, current_player)
                action_records[current_player.name].append((action, amount if amount else 0))
                
                if not game.handle_action(current_player, action, amount):
                    game.handle_action(current_player, 'fold', None)

        # === 结算 ===
        final_profit = players[0].stack - initial_stack
        
        # 计算最终奖励 (使用 SimpleRewardCalculator)
        # 注意: SimpleRewardCalculator.calculate_reward 参数顺序
        # (player, game, action, amount, is_winner, profit_loss, done)
        is_winner = (game.winner.name == "Main_AI") if hasattr(game, 'winner') and game.winner else False
        
        # 训练结束时的最终学习
        if is_training and last_step_info is not None:
            prev_state, prev_action_idx = last_step_info
            
            # 使用计算器计算最终奖励
            final_reward = self.reward_calculator.calculate_reward(
                player=players[0],
                game=game,
                action="end", # 占位符
                action_amount=0,
                is_winner=is_winner,
                profit_loss=final_profit,
                done=True
            )
            
            terminal_state = np.zeros_like(prev_state)
            self.main_agent.learn(final_reward, prev_state, prev_action_idx, terminal_state, True)

        result = {
            'winner': game.winner.name if hasattr(game, 'winner') and game.winner else 'Unknown',
            'player_profits': {p.name: p.stack - self.initial_stack for p in players},
            'action_histories': action_records
        }
        return result
    
    def evaluate(self) -> Dict:
        """评估模型性能"""
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
            
            main_actions = result['action_histories']['Main_AI']
            if main_actions:
                metrics = self._count_episode_metrics(players[0], game, main_actions)
                eval_bluffs += metrics['bluffs']
                eval_aggressive += metrics['aggressive']

        avg_profit = total_profit / self.num_eval_games
        win_rate = wins / self.num_eval_games
        bluff_rate = eval_bluffs / eval_aggressive if eval_aggressive > 0 else 0.0
        
        eval_result = {
            'avg_profit': avg_profit, 
            'win_rate': win_rate,
            'total_profit': total_profit, 
            'wins': wins,
            'bluff_rate': bluff_rate, 
            'total_hands': self.num_eval_games
        }

        logger.info(
            f"评估完成: 平均利润={avg_profit:.2f}, 胜率={win_rate:.2%}, "
            f"诈唬率={bluff_rate:.3f}"
        )
        self.performance_history.append(eval_result)
        return eval_result

    def update_opponents(self) -> None:
        """更新对手代理（内存复制版）"""
        if not self.opponent_agents:
            return
        try:
            # 随机选择一个正常对手进行更新（不更新最后的 Fish Agent）
            # self.opponent_agents[:-1] 是排除最后一个
            normal_opponents = self.opponent_agents[:-1]
            if not normal_opponents: return

            target_agent = random.choice(normal_opponents)
            
            weights = self.main_agent.get_weights()
            if weights is not None:
                target_agent.set_weights(weights)
                logger.info("已将主代理权重同步给随机一个对手 (Fish Agent 除外)")
            else:
                logger.warning("主代理不支持权重获取，跳过对手更新")
        except Exception as e:
            logger.error(f"更新对手失败: {e}")

    def save_checkpoint(self, is_best: bool = False) -> None:
        """保存检查点"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if is_best:
                filename = f"best_agent_{self.model_type}_{timestamp}.pkl"
            else:
                filename = f"agent_{self.model_type}_{self.total_hands}.pkl"
            filepath = os.path.join(self.checkpoint_dir, filename)
            
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
            
            success = save_checkpoint(self.main_agent, filepath, metadata)
            if success:
                logger.info(f"保存检查点: {filename}")
            else:
                logger.warning("工具函数保存失败，使用后备方法")
                self.main_agent.save_model(filepath)
                # 兼容旧逻辑，顺便存个state文件
                state_path = filepath.replace('.pkl', '') + '_state.pkl'
                if filepath.endswith('.weights.h5'): # 处理特殊后缀
                     state_path = filepath.replace('.weights.h5', '_state.pkl')

                with open(state_path, 'wb') as f:
                    pickle.dump(metadata, f)
                logger.info(f"使用后备方法保存检查点: {filename}")
                
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
            logger.error(traceback.format_exc())

    def load_checkpoint(self, filepath: Optional[str] = None) -> bool:
        """加载检查点（优化健壮性版）"""
        if filepath is None:
            logger.warning("未指定检查点路径，不自动加载最新检查点")
            return False
        
        logger.info(f"🔍 [DEBUG] 准备加载检查点: {filepath}")
        
        try:
            # 1. 验证文件是否存在 (支持模糊匹配)
            real_path = filepath
            if not os.path.exists(real_path):
                if os.path.exists(real_path + '.weights.h5'):
                    real_path += '.weights.h5'
                elif os.path.exists(real_path + '.pkl'):
                    real_path += '.pkl'
                else:
                    logger.error(f"❌ [DEBUG] 无法找到文件: {filepath}")
                    return False
            
            # 2. 加载模型权重
            self.main_agent.load_model(real_path)
            logger.info("✅ [DEBUG] 模型权重加载成功")

            # 3. 加载训练状态 (state)
            # 智能推断 state 文件路径
            state_path = None
            if real_path.endswith('.pkl'):
                state_path = real_path.replace('.pkl', '_state.pkl')
            elif real_path.endswith('.weights.h5'):
                state_path = real_path.replace('.weights.h5', '_state.pkl')
            
            if state_path and os.path.exists(state_path):
                try:
                    with open(state_path, 'rb') as f:
                        state = pickle.load(f)
                        self.total_hands = state.get('total_hands', 0)
                        self.total_profit = state.get('total_profit', 0)
                        self.best_score = state.get('best_score', -float('inf'))
                        self.performance_history = state.get('performance_history', [])
                        logger.info("✅ [DEBUG] 训练状态加载成功")
                except Exception as e:
                    logger.warning(f"⚠️ [DEBUG] 状态文件加载失败: {e}")
            
            return True

        except Exception as e:
            logger.error(f"❌ [DEBUG] 加载检查点严重错误: {e}")
            logger.error(traceback.format_exc()) 
            return False

    def record_performance(self, profit_loss: float) -> None:
        """记录训练表现"""
        # 计算策略指标
        bluff_rate = self.total_bluff_attempts / self.total_aggressive_actions if self.total_aggressive_actions > 0 else 0.0
        invalid_bet_rate = self.total_invalid_bets / self.total_raise_actions if self.total_raise_actions > 0 else 0.0
        
        # 详细日志记录
        logger.info(
            f"训练统计 - 总手牌: {self.total_hands}, "
            f"累计利润: {self.total_profit:.2f}, "
            f"当前手牌利润: {profit_loss:.2f}, "
            f"诈唬率: {bluff_rate:.3f}, "
            f"无效下注率: {invalid_bet_rate:.3f}"
        )
        
        # 为所有奖励计算器类型提供兼容支持
        # 如果是RewardCalculator，记录额外的性能指标
        if hasattr(self.reward_calculator, 'record_performance'):
            try:
                self.reward_calculator.record_performance(
                    profit_loss=profit_loss,
                    hands_played=1,
                    bluff_rate=bluff_rate,
                    invalid_bet_rate=invalid_bet_rate
                )
            except Exception as e:
                logger.warning(f"奖励计算器记录性能失败: {e}")
        
        # 记录到性能历史，方便后续分析
        performance_data = {
            'hands': self.total_hands,
            'total_profit': self.total_profit,
            'current_profit': profit_loss,
            'bluff_rate': bluff_rate,
            'invalid_bet_rate': invalid_bet_rate
        }
        logger.debug(f"性能数据: {performance_data}")

    def train(self, num_hands: int) -> None:
        """主训练循环"""
        logger.info(f"开始训练，共{num_hands}手牌")
        logger.info("从当前模型状态开始训练")
        start_hand = self.total_hands

        for _ in range(start_hand, start_hand + num_hands):
            game, players = self._create_game()
            result = self._play_hand(game, players, is_training=True)
            
            self.total_hands += 1
            self.total_profit += result['player_profits']['Main_AI']
            
            if self.total_hands % 200 == 0:
               self.record_performance(result['player_profits']['Main_AI'])
            
            if self.total_hands % self.eval_interval == 0:
                eval_result = self.evaluate()
                current_score = eval_result['avg_profit'] * eval_result['win_rate']
                # 只有当分数显著提高时才保存 Best
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.save_checkpoint(is_best=True)
                    logger.info(f"新最佳模型！得分: {current_score:.4f}")
            
            if self.total_hands % self.opponent_update_interval == 0:
                self.update_opponents()
            
            if self.total_hands % self.checkpoint_interval == 0:
                self.save_checkpoint(is_best=False)
        
        logger.info(f"训练完成！总手牌: {self.total_hands}")
        self.save_checkpoint(is_best=False)
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
    # 记得：如果是新训练，请先清理 checkpoints 目录
    # trainer.train(num_hands=100)