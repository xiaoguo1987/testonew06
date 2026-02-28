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
                 batch_size: int = 64,
                 optimizer: str = 'adam',
                 target_update_freq: int = 100,
                 ppo_clip_epsilon: float = 0.2,
                 ppo_update_epochs: int = 10,
                 ppo_entropy_coef: float = 0.02,
                 skip_final_eval: bool = False,
                 allin_unlucky_eq_threshold: float = 0.65,
                 allin_unlucky_bonus_cap: float = 0.80):
        
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
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.ppo_clip_epsilon = ppo_clip_epsilon
        self.ppo_update_epochs = ppo_update_epochs
        self.ppo_entropy_coef = ppo_entropy_coef
        self.skip_final_eval = skip_final_eval
        self.allin_unlucky_eq_threshold = allin_unlucky_eq_threshold
        self.allin_unlucky_bonus_cap = allin_unlucky_bonus_cap

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
        # 牌桌创建计数：用于让每手牌的庄位轮转，避免位置长期固定
        self._table_hand_counter = 0
        # 训练期每隔固定手数做一次精确策略统计，避免“训练全0、评估很高”的口径偏差
        self._exact_metrics_interval = 200
        # best 选择稳定性参数
        self.best_invalid_bet_threshold = 0.15
        self.best_score_window = 3
        # all-in 不走运补偿统计（用于监控补偿是否过量）
        self.allin_unlucky_trigger_count = 0
        self.allin_unlucky_bonus_total = 0.0
        self.allin_unlucky_equity_sum = 0.0
    
    def _create_agents(self) -> None:
        """创建AI代理（主代理+对手代理）"""
        if self.use_ensemble:
            self.main_agent = EnsembleAgent()
        else:
            kwargs = dict(
                agent_type=self.model_type,
                target_update_freq=self.target_update_freq,
                learning_rate=self.learning_rate,
                optimizer=self.optimizer,
            )
            if self.model_type == 'ppo':
                kwargs['ppo_clip_epsilon'] = self.ppo_clip_epsilon
                kwargs['ppo_update_epochs'] = self.ppo_update_epochs
                kwargs['ppo_entropy_coef'] = self.ppo_entropy_coef
                kwargs['ppo_batch_size'] = self.batch_size
            self.main_agent = PokerAIAgent(**kwargs)

        self.opponent_agents = []
        for i in range(self.num_players - 2):
            opp_kw = dict(
                agent_type=self.model_type,
                target_update_freq=self.target_update_freq,
                learning_rate=self.learning_rate,
                optimizer=self.optimizer,
            )
            if self.model_type == 'ppo':
                opp_kw['ppo_clip_epsilon'] = self.ppo_clip_epsilon
                opp_kw['ppo_update_epochs'] = self.ppo_update_epochs
                opp_kw['ppo_entropy_coef'] = self.ppo_entropy_coef
                opp_kw['ppo_batch_size'] = self.batch_size
            self.opponent_agents.append(PokerAIAgent(**opp_kw))
            
        # 第 5 个对手作为“鱼” (Random Player)
        # 对 PPO 训练场景，鱼玩家改为随机 DQN，避免“鱼”也变成 PPO 策略体
        fish_agent_type = 'dqn' if self.model_type == 'ppo' else self.model_type
        fish_agent = PokerAIAgent(agent_type=fish_agent_type)
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
        game = PokerGame(self.small_blind, self.big_blind, players)
        # 注意：start_new_hand 会先 dealer_pos + 1，因此这里预置“上一手庄位”
        game.dealer_pos = self._table_hand_counter % len(players)
        self._table_hand_counter += 1
        return game, players
    
    def _count_episode_metrics(self, main_player: Player, game: PokerGame,
                              action_history: List[Tuple[str, int]],
                              lightweight: bool = False) -> Dict[str, int]:
        """统计单局策略指标"""
        bluffs = 0
        aggressive = 0
        raises = 0
        invalid_bets = 0

        # --- 修正2: 只有当 reward_calculator 支持复杂逻辑时才计算 ---
        # SimpleRewardCalculator 是极简版，可能没有 is_bluffing 等方法
        # 这里做一个简单的兼容性检查
        has_complex_logic = (not lightweight) and hasattr(self.reward_calculator, 'is_bluffing')

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
        # 记录主AI是否在前两条街（翻牌前/翻牌）发生 all-in 以及当时估计胜率
        all_in_context = {
            'happened': False,
            'round': None,
            'equity': 0.0
        }

        while not game.is_hand_over():
            if game.is_round_over():
                if not game.proceed_to_next_round():
                    break
            
            current_player = game.next_player()
            if not current_player or not current_player.is_active:
                continue

            # === 主AI 行动逻辑 ===
            if current_player.name == "Main_AI":
                use_ppo_trajectory = getattr(self.main_agent, 'agent_type', None) == 'ppo'

                if is_training and not use_ppo_trajectory and last_step_info is not None:
                    current_state = self.main_agent.state_rep.get_state(game, current_player)
                    prev_state, prev_action_idx = last_step_info
                    self.main_agent.learn(0.001, prev_state, prev_action_idx, current_state, False)

                action_str, amount, action_info = self.main_agent.select_action_with_info(game, current_player)
                action_idx = action_info.get('action_idx', 0)
                current_state = action_info.get(
                    'state',
                    self.main_agent.state_rep.get_state(game, current_player),
                )

                action_records[current_player.name].append((action_str, amount if amount else 0))
                success = game.handle_action(current_player, action_str, amount)
                # 只在首次 all-in 时记录，避免重复覆盖
                if success and (not all_in_context['happened']) and current_player.is_all_in and game.round in (1, 2):
                    try:
                        # 使用状态表示的快速估计，减少训练路径开销
                        eq = float(self.main_agent.state_rep.calculate_hand_strength(current_player, game))
                    except Exception:
                        eq = 0.0
                    all_in_context.update({
                        'happened': True,
                        'round': int(game.round),
                        'equity': max(0.0, min(1.0, eq))
                    })

                if not success:
                    game.handle_action(current_player, 'fold', None)
                    if is_training and use_ppo_trajectory:
                        probs = action_info.get('action_probabilities', None)
                        fold_prob = float(np.clip(probs[0], 1e-8, 1.0)) if probs is not None else 1.0
                        # 动作非法时真实执行为 fold，轨迹记录也必须是 fold
                        self.main_agent.append_step(current_state, 0, fold_prob)
                    elif is_training and not use_ppo_trajectory and last_step_info is not None:
                        self.main_agent.learn(
                            reward=-0.05,
                            state=last_step_info[0], action=last_step_info[1],
                            next_state=current_state, done=False
                        )
                elif is_training and use_ppo_trajectory:
                    # 动作合法时，记录真实执行动作
                    old_prob = float(action_info.get('old_prob', 1.0))
                    self.main_agent.append_step(current_state, int(action_idx), old_prob)
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

        # 若只剩一个活跃玩家且底池未派发，需在这里派奖，避免利润长期为0
        alive = [p for p in players if p.is_active]
        if len(alive) == 1 and game.pot > 0:
            game._award_pot(alive[0])

        # === 结算 ===
        final_profit = players[0].stack - initial_stack
        
        # 计算最终奖励 (使用 SimpleRewardCalculator)
        # 注意: SimpleRewardCalculator.calculate_reward 参数顺序
        # (player, game, action, amount, is_winner, profit_loss, done)
        winners = [p.name for p in getattr(game, 'winners', [])]
        if not winners and hasattr(game, 'winner') and game.winner:
            winners = [game.winner.name]
        is_winner = "Main_AI" in winners
        
        # 训练结束时的最终学习
        if is_training and last_step_info is not None:
            final_reward = self.reward_calculator.calculate_reward(
                player=players[0],
                game=game,
                action="end",
                action_amount=0,
                is_winner=is_winner,
                profit_loss=final_profit,
                done=True
            )
            # 轻量反塌缩 shaping：若整手牌只有 fold 且直接输盲注，额外小惩罚，避免策略塌缩为“总弃牌”
            main_actions = action_records.get("Main_AI", [])
            if len(main_actions) == 1 and main_actions[0][0] == "fold" and final_profit <= 0:
                final_reward -= 0.25
            # all-in 不走运补偿：若在翻牌前/翻牌以高胜率打光，最终输掉，给予少量正向补偿
            # 目的：避免模型因短期方差而过度惩罚“长期EV正确”的激进行为
            if (not is_winner) and all_in_context['happened'] and all_in_context['round'] in (1, 2):
                eq = all_in_context['equity']
                if eq >= self.allin_unlucky_eq_threshold:
                    # 胜率越高补偿越高；封顶由配置控制
                    unlucky_bonus = min(
                        self.allin_unlucky_bonus_cap,
                        0.20 + (eq - self.allin_unlucky_eq_threshold) * 2.0
                    )
                    final_reward += unlucky_bonus
                    self.allin_unlucky_trigger_count += 1
                    self.allin_unlucky_bonus_total += float(unlucky_bonus)
                    self.allin_unlucky_equity_sum += float(eq)
            use_ppo_trajectory = getattr(self.main_agent, 'agent_type', None) == 'ppo'
            if use_ppo_trajectory:
                self.main_agent.finish_trajectory(final_reward)
            else:
                prev_state, prev_action_idx = last_step_info
                terminal_state = np.zeros_like(prev_state)
                self.main_agent.learn(final_reward, prev_state, prev_action_idx, terminal_state, True)

        result = {
            'winner': game.winner.name if hasattr(game, 'winner') and game.winner else 'Unknown',
            'winners': winners,
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
        eval_raises = 0
        eval_invalid_bets = 0

        for eval_idx in range(self.num_eval_games):
            # 每10局评估记录一次进度
            if (eval_idx + 1) % 10 == 0:
                logger.debug(f"评估进度: {eval_idx + 1}/{self.num_eval_games} 局")
                
            game, players = self._create_game()
            result = self._play_hand(game, players, is_training=False)
            total_profit += result['player_profits']['Main_AI']
            if 'Main_AI' in result.get('winners', []):
                wins += 1
            
            main_actions = result['action_histories']['Main_AI']
            if main_actions:
                metrics = self._count_episode_metrics(players[0], game, main_actions)
                eval_bluffs += metrics['bluffs']
                eval_aggressive += metrics['aggressive']
                eval_raises += metrics['raises']
                eval_invalid_bets += metrics['invalid_bets']

        avg_profit = total_profit / self.num_eval_games
        win_rate = wins / self.num_eval_games
        bluff_rate = eval_bluffs / eval_aggressive if eval_aggressive > 0 else 0.0
        raise_rate = eval_raises / (eval_aggressive + 1e-8) if eval_aggressive > 0 else 0.0
        invalid_bet_rate = eval_invalid_bets / (eval_raises + 1e-8) if eval_raises > 0 else 0.0
        
        eval_result = {
            'avg_profit': avg_profit, 
            'win_rate': win_rate,
            'total_profit': total_profit, 
            'wins': wins,
            'bluff_rate': bluff_rate, 
            'raise_rate': raise_rate,
            'invalid_bet_rate': invalid_bet_rate,
            'total_hands': self.num_eval_games
        }

        logger.info(
            f"评估完成: 平均利润={avg_profit:.2f}, 胜率={win_rate:.2%}, "
            f"诈唬率={bluff_rate:.3f}, 加注率={raise_rate:.3f}, 无效下注率={invalid_bet_rate:.3f}"
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
        """保存检查点。PPO 使用 base 路径（无 .pkl），存 .actor.weights.h5 / .critic.weights.h5 / _state.pkl"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            is_ppo = self.model_type == 'ppo'
            if is_ppo:
                base_name = f"best_agent_{self.model_type}_{timestamp}" if is_best else f"agent_{self.model_type}_{self.total_hands}"
            else:
                base_name = f"best_agent_{self.model_type}_{timestamp}.pkl" if is_best else f"agent_{self.model_type}_{self.total_hands}.pkl"
            filepath = os.path.join(self.checkpoint_dir, base_name)

            metadata = {
                'total_hands': self.total_hands,
                'total_profit': self.total_profit,
                'best_score': self.best_score,
                'performance_history': self.performance_history,
                'strategy_metrics': {
                    'bluff_attempts': self.total_bluff_attempts,
                    'aggressive_actions': self.total_aggressive_actions,
                    'allin_unlucky_trigger_count': self.allin_unlucky_trigger_count,
                    'allin_unlucky_bonus_total': self.allin_unlucky_bonus_total,
                    'allin_unlucky_equity_sum': self.allin_unlucky_equity_sum
                },
                'model_type': self.model_type
            }

            if is_ppo:
                self.main_agent.save_model(filepath)
                state_path = filepath + '_state.pkl'
                with open(state_path, 'wb') as f:
                    pickle.dump(metadata, f)
                logger.info(f"保存 PPO 检查点: {base_name}.actor.weights.h5 / .critic.weights.h5")
            else:
                success = save_checkpoint(self.main_agent, filepath, metadata)
                if success:
                    logger.info(f"保存检查点: {base_name}")
                else:
                    self.main_agent.save_model(filepath)
                    state_path = filepath.replace('.pkl', '') + '_state.pkl'
                    if filepath.endswith('.weights.h5'):
                        state_path = filepath.replace('.weights.h5', '_state.pkl')
                    with open(state_path, 'wb') as f:
                        pickle.dump(metadata, f)
                    logger.info(f"使用后备方法保存检查点: {base_name}")
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
            logger.error(traceback.format_exc())

    def _find_latest_checkpoint(self) -> Optional[str]:
        """查找最新的检查点文件。PPO 按 .actor.weights.h5 找并返回 base 路径。"""
        if not os.path.exists(self.checkpoint_dir):
            logger.warning(f"检查点目录不存在: {self.checkpoint_dir}")
            return None

        if self.model_type == 'ppo':
            candidates = []
            for file in os.listdir(self.checkpoint_dir):
                if file.endswith('.actor.weights.h5'):
                    base = os.path.join(self.checkpoint_dir, file.replace('.actor.weights.h5', ''))
                    candidates.append((os.path.getmtime(base + '.actor.weights.h5'), base))
            if not candidates:
                logger.warning(f"在 {self.checkpoint_dir} 中未找到 PPO 检查点 (.actor.weights.h5)")
                return None
            latest_file = max(candidates, key=lambda x: x[0])[1]
        else:
            checkpoint_files = []
            for file in os.listdir(self.checkpoint_dir):
                if (file.endswith('.weights.h5') or file.endswith('.pkl')) and not file.endswith('_state.pkl'):
                    checkpoint_files.append(os.path.join(self.checkpoint_dir, file))
            if not checkpoint_files:
                logger.warning(f"在 {self.checkpoint_dir} 中未找到检查点文件")
                return None
            latest_file = max(checkpoint_files, key=os.path.getmtime)
        logger.info(f"找到最新的检查点文件: {latest_file}")
        return latest_file
    
    def load_checkpoint(self, filepath: Optional[str] = None) -> bool:
        """加载检查点（优化健壮性版）"""
        if filepath is None:
            logger.info("未指定检查点路径，尝试自动查找最新检查点...")
            filepath = self._find_latest_checkpoint()
            if filepath is None:
                return False
        
        logger.info(f"🔍 [DEBUG] 准备加载检查点: {filepath}")
        
        try:
            real_path = filepath
            if self.model_type == 'ppo':
                if not real_path.endswith('.actor.weights.h5') and not real_path.endswith('.critic.weights.h5'):
                    if not os.path.exists(real_path + '.actor.weights.h5'):
                        logger.error(f"❌ [DEBUG] 无法找到 PPO 权重: {filepath}.actor.weights.h5")
                        return False
            elif not os.path.exists(real_path):
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
            state_path = None
            if self.model_type == 'ppo':
                state_path = real_path + '_state.pkl' if not real_path.endswith('_state.pkl') else real_path
            elif real_path.endswith('.pkl'):
                state_path = real_path.replace('.pkl', '_state.pkl')
            elif real_path.endswith('.weights.h5'):
                state_path = real_path.replace('.weights.h5', '_state.pkl')
            else:
                state_path = real_path + '_state.pkl'
            
            if state_path and os.path.exists(state_path):
                try:
                    with open(state_path, 'rb') as f:
                        state = pickle.load(f)
                        self.total_hands = state.get('total_hands', 0)
                        self.total_profit = state.get('total_profit', 0)
                        self.best_score = state.get('best_score', -float('inf'))
                        self.performance_history = state.get('performance_history', [])
                        self._table_hand_counter = self.total_hands
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
        if self.allin_unlucky_trigger_count > 0:
            avg_allin_bonus = self.allin_unlucky_bonus_total / self.allin_unlucky_trigger_count
            avg_allin_equity = self.allin_unlucky_equity_sum / self.allin_unlucky_trigger_count
        else:
            avg_allin_bonus = 0.0
            avg_allin_equity = 0.0
        
        # 详细日志记录
        logger.info(
            f"训练统计 - 总手牌: {self.total_hands}, "
            f"累计利润: {self.total_profit:.2f}, "
            f"当前手牌利润: {profit_loss:.2f}, "
            f"诈唬率: {bluff_rate:.3f}, "
            f"无效下注率: {invalid_bet_rate:.3f}, "
            f"allin补偿触发: {self.allin_unlucky_trigger_count}, "
            f"allin补偿均值: {avg_allin_bonus:.3f}, "
            f"allin触发时平均胜率: {avg_allin_equity:.3f}"
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
        t_env_total = 0.0
        t_train_total = 0.0
        if self.model_type == "ppo" and hasattr(self.main_agent, "agent") and hasattr(self.main_agent.agent, "warmup"):
            self.main_agent.agent.warmup()

        for hand_idx in range(num_hands):
            game, players = self._create_game()
            t0 = datetime.now().timestamp()
            result = self._play_hand(game, players, is_training=True)
            t1 = datetime.now().timestamp()
            
            self.total_hands += 1
            self.total_profit += result['player_profits']['Main_AI']
            t_env_total += (t1 - t0)

            # 累计策略统计，修复日志中诈唬率/无效下注率长期为0的问题
            main_actions = result.get('action_histories', {}).get('Main_AI', [])
            if main_actions:
                # 大部分手牌用轻量统计保证速度；周期性做精确统计修正口径偏差
                exact_metrics = (self.total_hands % self._exact_metrics_interval == 0)
                metrics = self._count_episode_metrics(players[0], game, main_actions, lightweight=not exact_metrics)
                self.total_bluff_attempts += metrics.get('bluffs', 0)
                self.total_aggressive_actions += metrics.get('aggressive', 0)
                self.total_raise_actions += metrics.get('raises', 0)
                self.total_invalid_bets += metrics.get('invalid_bets', 0)
            
            current_hand = start_hand + hand_idx + 1
            progress = ((hand_idx + 1) / num_hands) * 100
            
            if self.total_hands % 50 == 0:
               self.record_performance(result['player_profits']['Main_AI'])
            
            # 添加训练进度日志（每100手牌）
            if (hand_idx + 1) % 100 == 0:
                logger.info(f"训练进度: {current_hand}/{start_hand + num_hands} 手牌 ({progress:.1f}%)")
            
            # 记录详细手牌结果（每1000手牌）
            if (hand_idx + 1) % 1000 == 0:
                logger.debug(f"第 {current_hand} 手牌结果: {result}")
            
            if self.total_hands % self.eval_interval == 0:
                te0 = datetime.now().timestamp()
                eval_result = self.evaluate()
                te1 = datetime.now().timestamp()
                t_train_total += (te1 - te0)
                # 避免 avg_profit*win_rate 在 win_rate=0 时退化为 -0.0，导致“假最佳”
                # 综合收益、胜率和无效下注率，保持量纲可解释
                current_score = (
                    float(eval_result['avg_profit']) +
                    20.0 * float(eval_result['win_rate']) -
                    30.0 * float(eval_result.get('invalid_bet_rate', 0.0))
                )
                # 用最近N次评估分数做平滑，降低单点评估噪声对best选择的影响
                if not hasattr(self, "_best_score_history"):
                    self._best_score_history = []
                self._best_score_history.append(current_score)
                if len(self._best_score_history) > self.best_score_window:
                    self._best_score_history.pop(0)
                smoothed_score = float(np.mean(self._best_score_history))
                invalid_rate = float(eval_result.get('invalid_bet_rate', 0.0))
                # 加硬门槛：无效下注率过高时不允许当选best
                if invalid_rate <= self.best_invalid_bet_threshold and smoothed_score > self.best_score:
                    self.best_score = smoothed_score
                    self.save_checkpoint(is_best=True)
                    logger.info(
                        f"新最佳模型！平滑得分: {smoothed_score:.4f}, "
                        f"当前得分: {current_score:.4f}, 无效下注率: {invalid_rate:.3f}"
                    )
                elif invalid_rate > self.best_invalid_bet_threshold:
                    logger.info(
                        f"跳过最佳模型保存：无效下注率过高 ({invalid_rate:.3f} > {self.best_invalid_bet_threshold:.3f})"
                    )
            
            if self.total_hands % self.opponent_update_interval == 0:
                logger.info(f"更新对手模型，当前手牌数: {self.total_hands}")
                self.update_opponents()
            
            if self.total_hands % self.checkpoint_interval == 0:
                logger.info(f"保存检查点，当前手牌数: {self.total_hands}")
                self.save_checkpoint(is_best=False)
        
        logger.info(f"训练完成！总手牌: {self.total_hands}")
        self.save_checkpoint(is_best=False)
        if not self.skip_final_eval:
            te0 = datetime.now().timestamp()
            final_eval = self.evaluate()
            te1 = datetime.now().timestamp()
            t_train_total += (te1 - te0)
            logger.info(f"最终评估: {final_eval}")
        else:
            logger.info("已按配置跳过最终评估（--skip-final-eval）")
        if num_hands > 0:
            logger.info(
                f"性能统计: 环境与决策总耗时={t_env_total:.2f}s, 评估总耗时={t_train_total:.2f}s, "
                f"平均每手环境耗时={t_env_total/num_hands:.4f}s"
            )

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