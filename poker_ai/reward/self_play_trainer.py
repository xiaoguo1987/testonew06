# 德州扑克自我对弈训练器（优化版 - 最终修正）

import os
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
import pickle
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import threading

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SelfPlayTrainer')

from ..game.poker_game import PokerGame, Player
from ..model.ai_agent import PokerAIAgent, EnsembleAgent, GPUInferenceBridge, LightweightInferenceAgent
from ..model.ai_agent import _get_legal_action_mask
from ..model.ai_agent import _apply_preflop_position_filter
from ..model.state_representation import StateRepresentation
from ..reward.reward_calculator import RewardCalculator, AdaptiveRewardCalculator
from ..utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from ..reward.simple_reward_calculator import SimpleRewardCalculator


def _resolve_workers(requested_workers: int, reserve_cores: int = 2) -> int:
    """将 worker 数规范化为可用范围；<=0 时按 CPU 核数自动推断。"""
    if requested_workers > 0:
        return int(requested_workers)
    logical = os.cpu_count() or 4
    available = max(1, logical - max(0, reserve_cores))
    if available <= 4:
        return available
    return max(1, (available * 3) // 4)


def _round_down_power2(n: int) -> int:
    """将 n 向下取整到不大于 n 的 2 的幂（n<=0 返回 1）."""
    if n <= 0:
        return 1
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


class _AsyncPPOTrainer:
    """异步PPO训练器：双缓冲 + 后台训练线程。

    设计要点：
    - 主线程写入 buffer_b（采集轨迹），训练线程读取 buffer_a（训练数据）
    - 每隔一定批次，主线程 swap_buffers() 交换 a/b，唤醒训练线程
    - 训练线程完成后同步权重到 GPUInferenceBridge
    - 避免采样和训练串行，GPU 训练与 CPU 采样并行
    """

    def __init__(self, main_agent: PokerAIAgent, gpu_bridge: Optional[GPUInferenceBridge] = None):
        self.main_agent = main_agent
        self.gpu_bridge = gpu_bridge
        self._buffer_a = []  # 训练线程读
        self._buffer_b = []  # 主线程写
        self._lock = threading.Lock()
        self._train_event = threading.Event()
        self._stop_event = threading.Event()
        self._train_thread = threading.Thread(target=self._train_loop, daemon=True)
        self._train_thread.start()
        self._total_train_count = 0
        self._total_train_time = 0.0
        logger.info("[AsyncPPO] 异步训练线程已启动")

    def _train_loop(self):
        """后台训练循环"""
        while not self._stop_event.is_set():
            self._train_event.wait(timeout=1.0)
            self._train_event.clear()
            # 取出 buffer_a 的数据
            with self._lock:
                train_data = self._buffer_a
                self._buffer_a = []
            if not train_data:
                continue
            t0 = datetime.now().timestamp()
            try:
                # 批量写入轨迹并触发训练
                if hasattr(self.main_agent, 'append_trajectories_batch'):
                    self.main_agent.append_trajectories_batch(train_data)
                else:
                    for traj in train_data:
                        for s, a, p in zip(
                            traj.get('states', []),
                            traj.get('actions', []),
                            traj.get('old_probs', []),
                        ):
                            self.main_agent.append_step(s, a, p)
                        self.main_agent.finish_trajectory(traj.get('final_reward', 0.0))
                # 训练完成后同步权重到推理桥
                if self.gpu_bridge is not None and hasattr(self.main_agent.agent, 'get_weights_snapshot'):
                    snapshot = self.main_agent.agent.get_weights_snapshot()
                    self.gpu_bridge.sync_weights_from_main(snapshot)
            except Exception as e:
                err_msg = str(e).lower()
                if "dnn library" in err_msg or "graph execution" in err_msg:
                    logger.warning("[AsyncPPO] 训练跳过(DNN/Graph错误): %s", e)
                else:
                    logger.error("[AsyncPPO] 训练异常: %s", e, exc_info=True)
            elapsed = datetime.now().timestamp() - t0
            self._total_train_count += len(train_data)
            self._total_train_time += elapsed
            logger.info(
                f"[AsyncPPO] 训练批次完成: 轨迹数={len(train_data)}, "
                f"耗时={elapsed:.4f}s, 累计训练{self._total_train_count}条轨迹"
            )

    def submit_trajectories(self, trajectories):
        """主线程调用：将轨迹写入 buffer_b"""
        with self._lock:
            self._buffer_b.extend(trajectories)

    def swap_buffers(self):
        """主线程调用：交换 a/b 缓冲并唤醒训练线程"""
        with self._lock:
            self._buffer_a, self._buffer_b = self._buffer_b, self._buffer_a
        self._train_event.set()

    def pending_count(self) -> int:
        """当前 buffer_b 中待训练的轨迹数"""
        with self._lock:
            return len(self._buffer_b)

    def stop(self):
        """停止训练线程，先处理剩余数据"""
        # 先交换剩余数据
        with self._lock:
            if self._buffer_b:
                self._buffer_a, self._buffer_b = self._buffer_b, self._buffer_a
        self._train_event.set()
        self._stop_event.set()
        self._train_thread.join(timeout=15)
        if self._train_thread.is_alive():
            logger.warning("[AsyncPPO] 训练线程未能在15秒内结束")
        else:
            logger.info(
                f"[AsyncPPO] 训练线程已停止，累计训练{self._total_train_count}条轨迹，"
                f"总耗时{self._total_train_time:.2f}s"
            )

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
                 ppo_update_min_steps: int = 0,
                 train_rollouts_per_worker: int = 0,
                 train_sync_every_batches: int = 0,
                 dqn_replay_updates_per_hand: int = 0,
                 tuning_mode: bool = False,
                 skip_final_eval: bool = False,
                 allin_unlucky_eq_threshold: float = 0.65,
                 allin_unlucky_bonus_cap: float = 0.80,
                 eval_workers: int = 0,
                 train_workers: int = 0):
        
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
        self.ppo_update_min_steps = int(ppo_update_min_steps)
        self.tuning_mode = bool(tuning_mode)
        self.skip_final_eval = skip_final_eval
        self.allin_unlucky_eq_threshold = allin_unlucky_eq_threshold
        self.allin_unlucky_bonus_cap = allin_unlucky_bonus_cap
        self.eval_workers = _resolve_workers(int(eval_workers), reserve_cores=1)
        self.train_workers = _resolve_workers(int(train_workers), reserve_cores=2)
        if int(train_rollouts_per_worker) > 0:
            self.train_rollouts_per_worker = int(train_rollouts_per_worker)
        else:
            # PPO 训练更依赖大批量轨迹，默认给更大的并行块，降低调度开销
            if self.model_type == 'ppo':
                self.train_rollouts_per_worker = 8 if self.train_workers >= 32 else 4
            else:
                self.train_rollouts_per_worker = 2
        if int(train_sync_every_batches) > 0:
            self.train_sync_every_batches = int(train_sync_every_batches)
        else:
            if self.model_type == 'ppo':
                raw = max(12, self.train_workers // 2)
                self.train_sync_every_batches = max(12, _round_down_power2(raw))
            else:
                self.train_sync_every_batches = max(4, self.train_workers * 2)
        if int(dqn_replay_updates_per_hand) > 0:
            self.dqn_replay_updates_per_hand = int(dqn_replay_updates_per_hand)
        else:
            self.dqn_replay_updates_per_hand = 1

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
        self._opponents_version = 0
        self._last_regular_checkpoint_hands = None
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
                kwargs['ppo_update_min_steps'] = self.ppo_update_min_steps
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
                opp_kw['ppo_update_min_steps'] = self.ppo_update_min_steps
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
        # 并行场景下避免全局锁竞争：庄位随机化可保证长期公平性
        game.dealer_pos = random.randrange(len(players))
        return game, players
    
    def _count_episode_metrics(self, main_player: Optional[Player], game: Optional[PokerGame],
                              action_history: List[Tuple[str, int]],
                              lightweight: bool = False) -> Dict[str, int]:
        """统计单局策略指标。main_player/game 为 None 时自动退化为轻量统计。"""
        bluffs = 0
        aggressive = 0
        raises = 0
        invalid_bets = 0
        if main_player is None or game is None:
            lightweight = True
        # --- 修正2: 只有当 reward_calculator 支持复杂逻辑时才计算 ---
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
                    action_idx = 0
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
            'action_histories': action_records,
            'metrics': self._count_episode_metrics(
                players[0],
                game,
                action_records.get("Main_AI", []),
                lightweight=False
            )
        }
        return result

    def _play_hand_collect(
        self,
        game: PokerGame,
        players: List[Player],
        agent_copy: "PokerAIAgent",
        opponent_agents: Optional[List["PokerAIAgent"]] = None
    ) -> Tuple[Dict, Dict]:
        """
        仅做采样：用 agent_copy 选动作，不写 main_agent，返回 (result_dict, trajectory_data)。
        供并行 worker 使用；主线程收到后再对 main_agent 做 learn/append_step/finish_trajectory。
        """
        game.start_new_hand()
        last_step_info = None
        initial_stack = players[0].stack
        action_records = {p.name: [] for p in players}
        all_in_context = {'happened': False, 'round': None, 'equity': 0.0}
        opp_agents = opponent_agents if opponent_agents is not None else self.opponent_agents
        use_ppo = getattr(agent_copy, 'agent_type', None) == 'ppo'
        trajectory = (
            {'states': [], 'actions': [], 'old_probs': []}
            if use_ppo
            else {'transitions': []}
        )

        while not game.is_hand_over():
            if game.is_round_over():
                if not game.proceed_to_next_round():
                    break
            current_player = game.next_player()
            if not current_player or not current_player.is_active:
                continue

            if current_player.name == "Main_AI":
                if not use_ppo and last_step_info is not None:
                    current_state = agent_copy.state_rep.get_state(game, current_player)
                    prev_state, prev_action_idx = last_step_info
                    trajectory['transitions'].append(
                        (prev_state, prev_action_idx, 0.001, current_state, False)
                    )

                action_str, amount, action_info = agent_copy.select_action_with_info(
                    game, current_player
                )
                action_idx = action_info.get('action_idx', 0)
                current_state = action_info.get(
                    'state',
                    agent_copy.state_rep.get_state(game, current_player),
                )
                action_records[current_player.name].append(
                    (action_str, amount if amount else 0)
                )
                success = game.handle_action(current_player, action_str, amount)
                if success and (not all_in_context['happened']) and current_player.is_all_in and game.round in (1, 2):
                    try:
                        eq = float(
                            agent_copy.state_rep.calculate_hand_strength(
                                current_player, game
                            )
                        )
                    except Exception:
                        eq = 0.0
                    all_in_context.update({
                        'happened': True,
                        'round': int(game.round),
                        'equity': max(0.0, min(1.0, eq)),
                    })

                if not success:
                    game.handle_action(current_player, 'fold', None)
                    action_idx = 0
                    if use_ppo:
                        probs = action_info.get('action_probabilities', None)
                        fold_prob = (
                            float(np.clip(probs[0], 1e-8, 1.0)) if probs is not None
                            else 1.0
                        )
                        trajectory['states'].append(current_state)
                        trajectory['actions'].append(0)
                        trajectory['old_probs'].append(fold_prob)
                    else:
                        if last_step_info is not None:
                            prev_state, prev_action_idx = last_step_info
                            trajectory['transitions'].append(
                                (prev_state, prev_action_idx, -0.05, current_state, False)
                            )
                else:
                    if use_ppo:
                        old_prob = float(action_info.get('old_prob', 1.0))
                        trajectory['states'].append(current_state)
                        trajectory['actions'].append(int(action_idx))
                        trajectory['old_probs'].append(old_prob)
                last_step_info = (current_state, action_idx)
            else:
                try:
                    opp_idx = int(current_player.name.split('_')[-1]) - 1
                    if 0 <= opp_idx < len(opp_agents):
                        agent = opp_agents[opp_idx]
                    else:
                        agent = opp_agents[0]
                except Exception:
                    agent = opp_agents[0]
                action, amount = agent.select_action(game, current_player)
                action_records[current_player.name].append(
                    (action, amount if amount else 0)
                )
                if not game.handle_action(current_player, action, amount):
                    game.handle_action(current_player, 'fold', None)

        alive = [p for p in players if p.is_active]
        if len(alive) == 1 and game.pot > 0:
            game._award_pot(alive[0])

        final_profit = players[0].stack - initial_stack
        winners = [p.name for p in getattr(game, 'winners', [])]
        if not winners and hasattr(game, 'winner') and game.winner:
            winners = [game.winner.name]
        is_winner = "Main_AI" in winners

        final_reward = self.reward_calculator.calculate_reward(
            player=players[0],
            game=game,
            action="end",
            action_amount=0,
            is_winner=is_winner,
            profit_loss=final_profit,
            done=True,
        )
        main_actions = action_records.get("Main_AI", [])
        if len(main_actions) == 1 and main_actions[0][0] == "fold" and final_profit <= 0:
            final_reward -= 0.25
        allin_unlucky_bonus = 0.0
        if (not is_winner) and all_in_context['happened'] and all_in_context['round'] in (1, 2):
            eq = all_in_context['equity']
            if eq >= self.allin_unlucky_eq_threshold:
                unlucky_bonus = min(
                    self.allin_unlucky_bonus_cap,
                    0.20 + (eq - self.allin_unlucky_eq_threshold) * 2.0,
                )
                final_reward += unlucky_bonus
                allin_unlucky_bonus = unlucky_bonus
        trajectory['final_reward'] = final_reward
        trajectory['allin_unlucky'] = {
            'triggered': allin_unlucky_bonus > 0,
            'bonus': allin_unlucky_bonus,
            'equity': all_in_context['equity'],
        }

        if not use_ppo and last_step_info is not None:
            prev_state, prev_action_idx = last_step_info
            terminal_state = np.zeros_like(prev_state)
            trajectory['transitions'].append(
                (prev_state, prev_action_idx, final_reward, terminal_state, True)
            )

        result = {
            'winner': game.winner.name if hasattr(game, 'winner') and game.winner else 'Unknown',
            'winners': winners,
            'player_profits': {p.name: p.stack - self.initial_stack for p in players},
            'action_histories': action_records,
            'metrics': self._count_episode_metrics(
                players[0],
                game,
                action_records.get("Main_AI", []),
                lightweight=False
            ),
        }
        return result, trajectory

    def _play_hands_collect_batched(
        self,
        num_hands: int,
        agent_copy: "PokerAIAgent",
        opponent_agents: Optional[List["PokerAIAgent"]] = None,
    ) -> List[Tuple[Dict, Dict]]:
        """
        同时推进多局手牌，并对 Main_AI 使用批量推理。
        主要用于 PPO 并行采样，减少 Python/TensorFlow 往返开销。
        """
        if num_hands <= 0:
            return []

        opp_agents = opponent_agents if opponent_agents is not None else self.opponent_agents
        use_ppo = getattr(agent_copy, 'agent_type', None) == 'ppo'
        if not use_ppo or num_hands == 1:
            outputs = []
            for _ in range(num_hands):
                game, players = self._create_game()
                outputs.append(
                    self._play_hand_collect(
                        game,
                        players,
                        agent_copy,
                        opponent_agents=opp_agents,
                    )
                )
            return outputs

        envs = []
        for _ in range(num_hands):
            game, players = self._create_game()
            game.start_new_hand()
            envs.append({
                'game': game,
                'players': players,
                'last_step_info': None,
                'initial_stack': players[0].stack,
                'action_records': {p.name: [] for p in players},
                'all_in_context': {'happened': False, 'round': None, 'equity': 0.0},
                'trajectory': {'states': [], 'actions': [], 'old_probs': []},
                'done': False,
            })

        outputs: List[Tuple[Dict, Dict]] = []
        remaining = len(envs)

        def finalize_env(env_idx: int) -> None:
            env = envs[env_idx]
            if env['done']:
                return
            game = env['game']
            players = env['players']
            alive = [p for p in players if p.is_active]
            if len(alive) == 1 and game.pot > 0:
                game._award_pot(alive[0])

            final_profit = players[0].stack - env['initial_stack']
            winners = [p.name for p in getattr(game, 'winners', [])]
            if not winners and hasattr(game, 'winner') and game.winner:
                winners = [game.winner.name]
            is_winner = "Main_AI" in winners

            final_reward = self.reward_calculator.calculate_reward(
                player=players[0],
                game=game,
                action="end",
                action_amount=0,
                is_winner=is_winner,
                profit_loss=final_profit,
                done=True,
            )
            main_actions = env['action_records'].get("Main_AI", [])
            if len(main_actions) == 1 and main_actions[0][0] == "fold" and final_profit <= 0:
                final_reward -= 0.25
            allin_unlucky_bonus = 0.0
            all_in_context = env['all_in_context']
            if (not is_winner) and all_in_context['happened'] and all_in_context['round'] in (1, 2):
                eq = all_in_context['equity']
                if eq >= self.allin_unlucky_eq_threshold:
                    unlucky_bonus = min(
                        self.allin_unlucky_bonus_cap,
                        0.20 + (eq - self.allin_unlucky_eq_threshold) * 2.0,
                    )
                    final_reward += unlucky_bonus
                    allin_unlucky_bonus = unlucky_bonus

            env['trajectory']['final_reward'] = final_reward
            env['trajectory']['allin_unlucky'] = {
                'triggered': allin_unlucky_bonus > 0,
                'bonus': allin_unlucky_bonus,
                'equity': all_in_context['equity'],
            }

            result = {
                'winner': game.winner.name if hasattr(game, 'winner') and game.winner else 'Unknown',
                'winners': winners,
                'player_profits': {p.name: p.stack - self.initial_stack for p in players},
                'action_histories': env['action_records'],
                'metrics': self._count_episode_metrics(
                    players[0],
                    game,
                    env['action_records'].get("Main_AI", []),
                    lightweight=False,
                ),
            }
            outputs.append((result, env['trajectory']))
            env['done'] = True

        while remaining > 0:
            main_requests = []
            main_states = []
            progress_made = False

            for env_idx, env in enumerate(envs):
                if env['done']:
                    continue
                game = env['game']
                players = env['players']

                if game.is_hand_over():
                    finalize_env(env_idx)
                    remaining -= 1
                    progress_made = True
                    continue

                if game.is_round_over():
                    if game.proceed_to_next_round():
                        finalize_env(env_idx)
                        remaining -= 1
                    progress_made = True
                    continue

                current_player = game.next_player()
                if not current_player or not current_player.is_active:
                    continue

                if current_player.name == "Main_AI":
                    current_state = agent_copy.state_rep.get_state(game, current_player)
                    env['current_main_player'] = current_player
                    env['current_main_state'] = current_state
                    main_requests.append((game, current_player))
                    main_states.append(current_state)
                    progress_made = True
                else:
                    try:
                        opp_idx = int(current_player.name.split('_')[-1]) - 1
                        if 0 <= opp_idx < len(opp_agents):
                            agent = opp_agents[opp_idx]
                        else:
                            agent = opp_agents[0]
                    except Exception:
                        agent = opp_agents[0]
                    action, amount = agent.select_action(game, current_player)
                    env['action_records'][current_player.name].append(
                        (action, amount if amount else 0)
                    )
                    if not game.handle_action(current_player, action, amount):
                        game.handle_action(current_player, 'fold', None)
                    progress_made = True

            if main_requests:
                batched_actions = agent_copy.select_actions_batch_with_info(
                    main_requests,
                    precomputed_states=np.asarray(main_states, dtype=np.float32),
                )
                batched_idx = 0
                for env_idx, env in enumerate(envs):
                    if env['done'] or 'current_main_player' not in env:
                        continue
                    game = env['game']
                    current_player = env.pop('current_main_player')
                    current_state = env.pop('current_main_state')
                    action_str, amount, action_info = batched_actions[batched_idx]
                    batched_idx += 1
                    action_idx = action_info.get('action_idx', 0)
                    env['action_records'][current_player.name].append(
                        (action_str, amount if amount else 0)
                    )
                    success = game.handle_action(current_player, action_str, amount)
                    if success and (not env['all_in_context']['happened']) and current_player.is_all_in and game.round in (1, 2):
                        try:
                            eq = float(agent_copy.state_rep.calculate_hand_strength(current_player, game))
                        except Exception:
                            eq = 0.0
                        env['all_in_context'].update({
                            'happened': True,
                            'round': int(game.round),
                            'equity': max(0.0, min(1.0, eq)),
                        })

                    if not success:
                        game.handle_action(current_player, 'fold', None)
                        probs = action_info.get('action_probabilities', None)
                        fold_prob = float(np.clip(probs[0], 1e-8, 1.0)) if probs is not None else 1.0
                        env['trajectory']['states'].append(current_state)
                        env['trajectory']['actions'].append(0)
                        env['trajectory']['old_probs'].append(fold_prob)
                    else:
                        old_prob = float(action_info.get('old_prob', 1.0))
                        env['trajectory']['states'].append(current_state)
                        env['trajectory']['actions'].append(int(action_idx))
                        env['trajectory']['old_probs'].append(old_prob)
                    env['last_step_info'] = (current_state, action_idx)

            if not progress_made and not main_requests:
                # 理论上不应出现；若出现则避免死循环并强制结算剩余局面。
                for env_idx, env in enumerate(envs):
                    if not env['done']:
                        finalize_env(env_idx)
                        remaining -= 1

        return outputs

    def _play_hands_inference_batched(
        self,
        num_hands: int,
        main_agent: "PokerAIAgent",
        opponent_agents: Optional[List["PokerAIAgent"]] = None,
    ) -> List[Dict]:
        """纯推理版多局同步推进，对 Main_AI 使用批量推理。"""
        if num_hands <= 0:
            return []

        opp_agents = opponent_agents if opponent_agents is not None else self.opponent_agents
        if getattr(main_agent, 'agent_type', None) != 'ppo' or num_hands == 1:
            outputs = []
            for _ in range(num_hands):
                game, players = self._create_game()
                outputs.append(
                    self._play_hand_inference(
                        game,
                        players,
                        main_agent,
                        opponent_agents=opp_agents,
                    )
                )
            return outputs

        envs = []
        for _ in range(num_hands):
            game, players = self._create_game()
            game.start_new_hand()
            envs.append({
                'game': game,
                'players': players,
                'initial_stack': players[0].stack,
                'action_records': {p.name: [] for p in players},
                'done': False,
            })

        outputs: List[Dict] = []
        remaining = len(envs)

        def finalize_env(env_idx: int) -> None:
            env = envs[env_idx]
            if env['done']:
                return
            game = env['game']
            players = env['players']
            alive = [p for p in players if p.is_active]
            if len(alive) == 1 and game.pot > 0:
                game._award_pot(alive[0])
            winners = [p.name for p in getattr(game, 'winners', [])]
            if not winners and hasattr(game, 'winner') and game.winner:
                winners = [game.winner.name]
            outputs.append({
                'winner': game.winner.name if hasattr(game, 'winner') and game.winner else 'Unknown',
                'winners': winners,
                'player_profits': {p.name: p.stack - self.initial_stack for p in players},
                'action_histories': env['action_records'],
                'metrics': self._count_episode_metrics(
                    players[0],
                    game,
                    env['action_records'].get("Main_AI", []),
                    lightweight=False,
                ),
            })
            env['done'] = True

        while remaining > 0:
            main_requests = []
            main_states = []
            progress_made = False

            for env_idx, env in enumerate(envs):
                if env['done']:
                    continue
                game = env['game']

                if game.is_hand_over():
                    finalize_env(env_idx)
                    remaining -= 1
                    progress_made = True
                    continue

                if game.is_round_over():
                    if game.proceed_to_next_round():
                        finalize_env(env_idx)
                        remaining -= 1
                    progress_made = True
                    continue

                current_player = game.next_player()
                if not current_player or not current_player.is_active:
                    continue

                if current_player.name == "Main_AI":
                    env['current_main_player'] = current_player
                    main_requests.append((game, current_player))
                    main_states.append(main_agent.state_rep.get_state(game, current_player))
                    progress_made = True
                else:
                    try:
                        opp_idx = int(current_player.name.split('_')[-1]) - 1
                        if 0 <= opp_idx < len(opp_agents):
                            agent = opp_agents[opp_idx]
                        else:
                            agent = opp_agents[0]
                    except Exception:
                        agent = opp_agents[0]
                    action, amount = agent.select_action(game, current_player)
                    env['action_records'][current_player.name].append((action, amount if amount else 0))
                    if not game.handle_action(current_player, action, amount):
                        game.handle_action(current_player, 'fold', None)
                    progress_made = True

            if main_requests:
                batched_actions = main_agent.select_actions_batch_with_info(
                    main_requests,
                    precomputed_states=np.asarray(main_states, dtype=np.float32),
                )
                batched_idx = 0
                for env_idx, env in enumerate(envs):
                    if env['done'] or 'current_main_player' not in env:
                        continue
                    game = env['game']
                    current_player = env.pop('current_main_player')
                    action_str, amount, _ = batched_actions[batched_idx]
                    batched_idx += 1
                    env['action_records'][current_player.name].append(
                        (action_str, amount if amount else 0)
                    )
                    success = game.handle_action(current_player, action_str, amount)
                    if not success:
                        game.handle_action(current_player, 'fold', None)

            if not progress_made and not main_requests:
                for env_idx, env in enumerate(envs):
                    if not env['done']:
                        finalize_env(env_idx)
                        remaining -= 1

        return outputs

    def _play_hand_inference(
        self,
        game: PokerGame,
        players: List[Player],
        main_agent: "PokerAIAgent",
        opponent_agents: Optional[List["PokerAIAgent"]] = None
    ) -> Dict:
        """纯推理版单局，不写训练缓冲，供 evaluate 并行调用。"""
        game.start_new_hand()
        initial_stack = players[0].stack
        action_records = {p.name: [] for p in players}
        opp_agents = opponent_agents if opponent_agents is not None else self.opponent_agents

        while not game.is_hand_over():
            if game.is_round_over():
                if not game.proceed_to_next_round():
                    break
            current_player = game.next_player()
            if not current_player or not current_player.is_active:
                continue

            if current_player.name == "Main_AI":
                action_str, amount = main_agent.select_action(game, current_player)
                action_records[current_player.name].append((action_str, amount if amount else 0))
                success = game.handle_action(current_player, action_str, amount)
                if not success:
                    game.handle_action(current_player, 'fold', None)
            else:
                try:
                    opp_idx = int(current_player.name.split('_')[-1]) - 1
                    if 0 <= opp_idx < len(opp_agents):
                        agent = opp_agents[opp_idx]
                    else:
                        agent = opp_agents[0]
                except Exception:
                    agent = opp_agents[0]
                action, amount = agent.select_action(game, current_player)
                action_records[current_player.name].append((action, amount if amount else 0))
                if not game.handle_action(current_player, action, amount):
                    game.handle_action(current_player, 'fold', None)

        alive = [p for p in players if p.is_active]
        if len(alive) == 1 and game.pot > 0:
            game._award_pot(alive[0])

        winners = [p.name for p in getattr(game, 'winners', [])]
        if not winners and hasattr(game, 'winner') and game.winner:
            winners = [game.winner.name]
        return {
            'winner': game.winner.name if hasattr(game, 'winner') and game.winner else 'Unknown',
            'winners': winners,
            'player_profits': {p.name: p.stack - self.initial_stack for p in players},
            'action_histories': action_records,
            'metrics': self._count_episode_metrics(
                players[0],
                game,
                action_records.get("Main_AI", []),
                lightweight=False
            ),
        }

    def _clone_opponents_for_inference(self) -> List["PokerAIAgent"]:
        """为并行 worker 生成独立对手副本，避免共享模型并发推理争用。"""
        copies = []
        for opp in self.opponent_agents:
            if hasattr(opp, 'clone_for_inference'):
                copies.append(opp.clone_for_inference())
            else:
                copies.append(opp)
        return copies

    def _apply_trajectory(self, trajectory: Dict) -> None:
        """将单条轨迹应用到 main_agent（PPO: append_step + finish_trajectory；DQN: remember + replay）。"""
        use_ppo = getattr(self.main_agent, 'agent_type', None) == 'ppo'
        if use_ppo:
            for s, a, p in zip(
                trajectory['states'],
                trajectory['actions'],
                trajectory['old_probs'],
            ):
                self.main_agent.append_step(s, a, p)
            self.main_agent.finish_trajectory(trajectory['final_reward'])
        else:
            self.main_agent.learn_from_transitions(
                trajectory['transitions'],
                replay_updates=self.dqn_replay_updates_per_hand,
            )
    
    def evaluate(self) -> Dict:
        """评估模型性能"""
        logger.info(f"开始评估，共{self.num_eval_games}局")
        cpu_total = os.cpu_count() or 8
        effective_eval_workers = max(
            1,
            min(self.eval_workers, self.num_eval_games, max(1, min(24, cpu_total // 4)))
        )
        logger.info(
            f"评估并行配置: workers={effective_eval_workers} "
            f"(requested={self.eval_workers})"
        )
        total_profit = 0
        wins = 0
        eval_bluffs = 0
        eval_aggressive = 0
        eval_raises = 0
        eval_invalid_bets = 0

        # 在主线程预创建推理用克隆，避免在 worker 子线程创建 TF 模型导致 Variable shape 冲突（(64,5) vs (64,64)）
        if hasattr(self.main_agent, 'clone_for_inference'):
            eval_main_clones = [self.main_agent.clone_for_inference() for _ in range(effective_eval_workers)]
            eval_opp_clones_list = [self._clone_opponents_for_inference() for _ in range(effective_eval_workers)]
        else:
            eval_main_clones = [self.main_agent] * effective_eval_workers
            eval_opp_clones_list = [self.opponent_agents] * effective_eval_workers

        def _run_eval_chunk(chunk_size: int, main_clone: "PokerAIAgent", opp_clones: List["PokerAIAgent"]) -> List[Dict]:
            if chunk_size <= 0:
                return []
            if getattr(main_clone, 'agent_type', None) == 'ppo' and chunk_size > 1:
                results = self._play_hands_inference_batched(
                    chunk_size,
                    main_clone,
                    opp_clones,
                )
            else:
                results = []
                for _ in range(chunk_size):
                    game, players = self._create_game()
                    results.append(
                        self._play_hand_inference(
                            game,
                            players,
                            main_clone,
                            opponent_agents=opp_clones,
                        )
                    )
            payload = []
            for result in results:
                payload.append({
                    'profit': result['player_profits']['Main_AI'],
                    'win': 1 if 'Main_AI' in result.get('winners', []) else 0,
                    'metrics': result.get(
                        'metrics',
                        {'bluffs': 0, 'aggressive': 0, 'raises': 0, 'invalid_bets': 0}
                    ),
                    'worker_tid': threading.get_ident(),
                })
            return payload

        if effective_eval_workers > 1:
            worker_tids = set()
            base_chunk = self.num_eval_games // effective_eval_workers
            extra_chunk = self.num_eval_games % effective_eval_workers
            chunk_plan = [
                base_chunk + (1 if idx < extra_chunk else 0)
                for idx in range(effective_eval_workers)
                if base_chunk + (1 if idx < extra_chunk else 0) > 0
            ]
            with ThreadPoolExecutor(max_workers=effective_eval_workers) as executor:
                done_games = 0
                for chunk_items in executor.map(
                    _run_eval_chunk, chunk_plan, eval_main_clones, eval_opp_clones_list
                ):
                    for item in chunk_items:
                        total_profit += item['profit']
                        wins += item['win']
                        eval_bluffs += item['metrics']['bluffs']
                        eval_aggressive += item['metrics']['aggressive']
                        eval_raises += item['metrics']['raises']
                        eval_invalid_bets += item['metrics']['invalid_bets']
                        worker_tids.add(item['worker_tid'])
                        done_games += 1
                        if done_games % 10 == 0 or done_games == self.num_eval_games:
                            logger.info(f"评估进度: {done_games}/{self.num_eval_games} 局")
            logger.info(
                f"评估并行生效检查: 实际线程数={len(worker_tids)}, "
                f"配置线程数={effective_eval_workers}"
            )
        else:
            chunk_size = min(self.num_eval_games, max(1, min(16, self.train_rollouts_per_worker)))
            done_games = 0
            remaining_games = self.num_eval_games
            while remaining_games > 0:
                current_chunk = min(chunk_size, remaining_games)
                for item in _run_eval_chunk(
                    current_chunk, eval_main_clones[0], eval_opp_clones_list[0]
                ):
                    done_games += 1
                    total_profit += item['profit']
                    wins += item['win']
                    eval_bluffs += item['metrics']['bluffs']
                    eval_aggressive += item['metrics']['aggressive']
                    eval_raises += item['metrics']['raises']
                    eval_invalid_bets += item['metrics']['invalid_bets']
                    if done_games % 10 == 0 or done_games == self.num_eval_games:
                        logger.info(f"评估进度: {done_games}/{self.num_eval_games} 局")
                remaining_games -= current_chunk

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
                self._opponents_version += 1
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
                if not is_best:
                    self._last_regular_checkpoint_hands = self.total_hands
            else:
                success = save_checkpoint(self.main_agent, filepath, metadata)
                if success:
                    logger.info(f"保存检查点: {base_name}")
                    if not is_best:
                        self._last_regular_checkpoint_hands = self.total_hands
                else:
                    self.main_agent.save_model(filepath)
                    state_path = filepath.replace('.pkl', '') + '_state.pkl'
                    if filepath.endswith('.weights.h5'):
                        state_path = filepath.replace('.weights.h5', '_state.pkl')
                    with open(state_path, 'wb') as f:
                        pickle.dump(metadata, f)
                    logger.info(f"使用后备方法保存检查点: {base_name}")
                    if not is_best:
                        self._last_regular_checkpoint_hands = self.total_hands
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

    def _sync_copies_from_main(self, copies: List) -> None:
        """将 main_agent 的权重同步到各 inference 副本。"""
        weights = self.main_agent.get_weights()
        if weights is None:
            return
        for c in copies:
            c.set_weights(weights)
        if self.model_type == 'dqn' and hasattr(self.main_agent.agent, 'epsilon'):
            for c in copies:
                c.agent.epsilon = self.main_agent.agent.epsilon
                c.agent.epsilon_min = getattr(
                    self.main_agent.agent, 'epsilon_min', 0.1
                )
                c.agent.epsilon_decay = getattr(
                    self.main_agent.agent, 'epsilon_decay', 0.995
                )

    def train(self, num_hands: int) -> None:
        """主训练循环 - GPU流水线优化版。
        
        三维度优化：
        1. 更大模型（512-512-256-128）+更大batch → 更多GPU计算
        2. GPU批量推理桥 → 主线程统一GPU推理，worker只做环境模拟
        3. 异步训练线程 → 采样与训练并行，GPU不再空闲
        """
        logger.info(f"开始训练，共{num_hands}手牌")
        logger.info("从当前模型状态开始训练")
        logger.info(f"训练采样并行: workers={self.train_workers}")
        logger.info(f"训练并行批次: rollouts_per_worker={self.train_rollouts_per_worker}")
        logger.info(f"训练同步频率: sync_every_batches={self.train_sync_every_batches}")
        start_hand = self.total_hands
        t_env_total = 0.0
        t_train_total = 0.0
        t_gpu_infer_total = 0.0
        if self.model_type == "ppo" and hasattr(self.main_agent, "agent") and hasattr(self.main_agent.agent, "warmup"):
            self.main_agent.agent.warmup()

        use_parallel = (
            self.train_workers > 1
            and not self.use_ensemble
            and hasattr(self.main_agent, 'clone_for_inference')
        )

        # ===== 维度2+3: GPU批量推理桥 + 异步训练线程 =====
        use_gpu_pipeline = (
            self.model_type == 'ppo'
            and hasattr(self.main_agent, 'agent')
            and hasattr(self.main_agent.agent, 'predict_batch')
        )
        gpu_bridge = None
        async_trainer = None

        if use_gpu_pipeline:
            # 创建 GPU 推理桥
            gpu_bridge = GPUInferenceBridge(self.main_agent.agent)
            logger.info("[GPU流水线] 启用GPU批量推理桥 - 所有Main_AI决策统一在GPU上执行")
            # 创建异步训练器
            async_trainer = _AsyncPPOTrainer(self.main_agent, gpu_bridge)
            logger.info("[GPU流水线] 启用异步训练线程 - 采样与训练并行")

        # 为每个worker创建轻量级环境模拟Agent（只做状态编码，不做推理）
        # 同时保留对手Agent副本
        agent_copies = []
        worker_opp_copies = [None] * self.train_workers
        worker_opp_versions = [-1] * self.train_workers
        # 轻量级worker：只持有StateRepresentation，推理由GPU桥完成
        worker_state_reps = []

        if use_parallel:
            if use_gpu_pipeline:
                # GPU流水线模式：每个并发环境独立StateRepresentation，避免线程共享问题
                max_concurrent_envs = self.train_workers * max(1, self.train_rollouts_per_worker)
                for _ in range(max_concurrent_envs):
                    worker_state_reps.append(StateRepresentation())
                logger.info(
                    f"[GPU流水线] {max_concurrent_envs}个并发牌桌环境，"
                    f"每个独立状态编码器，推理统一由GPU执行"
                )
            else:
                agent_copies = [
                    self.main_agent.clone_for_inference()
                    for _ in range(self.train_workers)
                ]
                self._sync_copies_from_main(agent_copies)
            # 在主线程预创建对手副本
            for w in range(self.train_workers):
                worker_opp_copies[w] = self._clone_opponents_for_inference()
                worker_opp_versions[w] = self._opponents_version

        hand_idx = 0
        batch_counter = 0
        executor = ThreadPoolExecutor(max_workers=self.train_workers) if use_parallel else None
        try:
            while hand_idx < num_hands:
                remaining_hands = num_hands - hand_idx
                t0 = datetime.now().timestamp()

                if use_parallel and use_gpu_pipeline:
                    # ===== GPU流水线模式：迭代式环境推进 + GPU批量推理 =====
                    results_and_trajectories = self._gpu_pipeline_step(
                        executor, remaining_hands, worker_state_reps,
                        worker_opp_copies, gpu_bridge
                    )
                elif use_parallel:
                    # 原始并行模式（非PPO或未启用GPU流水线）
                    worker_count = min(self.train_workers, remaining_hands)
                    target_batch_hands = min(
                        remaining_hands,
                        worker_count * max(1, self.train_rollouts_per_worker)
                    )
                    base_rollouts = target_batch_hands // worker_count
                    extra_rollouts = target_batch_hands % worker_count
                    worker_plan = [
                        (idx, base_rollouts + (1 if idx < extra_rollouts else 0))
                        for idx in range(worker_count)
                    ]
                    if batch_counter % max(1, self.train_sync_every_batches) == 0:
                        self._sync_copies_from_main(agent_copies)

                    def run_one(task: Tuple[int, int]):
                        worker_idx, rollout_count = task
                        copy_agent = agent_copies[worker_idx]
                        return self._play_hands_collect_batched(
                            rollout_count,
                            copy_agent,
                            opponent_agents=worker_opp_copies[worker_idx]
                        )

                    results_and_trajectories = []
                    for res_list in executor.map(run_one, worker_plan):
                        results_and_trajectories.extend(res_list)
                else:
                    game, players = self._create_game()
                    result, trajectory = self._play_hand_collect(
                        game, players, self.main_agent
                    )
                    results_and_trajectories = [(result, trajectory)]

                t1 = datetime.now().timestamp()
                t_env_total += (t1 - t0)

                # 收集PPO轨迹
                ppo_trajectories = []
                for (result, trajectory) in results_and_trajectories:
                    if self.model_type != 'ppo':
                        self._apply_trajectory(trajectory)
                    else:
                        ppo_trajectories.append(trajectory)

                    self.total_hands += 1
                    self.total_profit += result['player_profits']['Main_AI']

                    au = trajectory.get('allin_unlucky', {})
                    if au.get('triggered'):
                        self.allin_unlucky_trigger_count += 1
                        self.allin_unlucky_bonus_total += float(au.get('bonus', 0))
                        self.allin_unlucky_equity_sum += float(au.get('equity', 0))

                    metrics = result.get('metrics')
                    if metrics:
                        self.total_bluff_attempts += metrics.get('bluffs', 0)
                        self.total_aggressive_actions += metrics.get('aggressive', 0)
                        self.total_raise_actions += metrics.get('raises', 0)
                        self.total_invalid_bets += metrics.get('invalid_bets', 0)

                    current_hand = self.total_hands
                    progress = (current_hand - start_hand) / num_hands * 100

                    if self.total_hands % 50 == 0:
                        self.record_performance(result['player_profits']['Main_AI'])

                    if (current_hand - start_hand) % 100 == 0:
                        logger.info(
                            f"训练进度: {current_hand}/{start_hand + num_hands} 手牌 ({progress:.1f}%)"
                        )
                    if (current_hand - start_hand) % 1000 == 0:
                        logger.debug(f"第 {current_hand} 手牌结果: {result}")

                    if (not self.tuning_mode) and self.total_hands % self.eval_interval == 0:
                        te0 = datetime.now().timestamp()
                        eval_result = self.evaluate()
                        te1 = datetime.now().timestamp()
                        t_train_total += (te1 - te0)
                        current_score = (
                            float(eval_result['avg_profit']) +
                            20.0 * float(eval_result['win_rate']) -
                            30.0 * float(eval_result.get('invalid_bet_rate', 0.0))
                        )
                        if not hasattr(self, "_best_score_history"):
                            self._best_score_history = []
                        self._best_score_history.append(current_score)
                        if len(self._best_score_history) > self.best_score_window:
                            self._best_score_history.pop(0)
                        smoothed_score = float(np.mean(self._best_score_history))
                        invalid_rate = float(eval_result.get('invalid_bet_rate', 0.0))
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

                    if (not self.tuning_mode) and self.total_hands % self.opponent_update_interval == 0:
                        logger.info(f"更新对手模型，当前手牌数: {self.total_hands}")
                        self.update_opponents()
                        if use_parallel:
                            for w in range(self.train_workers):
                                worker_opp_copies[w] = self._clone_opponents_for_inference()
                                worker_opp_versions[w] = self._opponents_version

                    if (not self.tuning_mode) and self.total_hands % self.checkpoint_interval == 0:
                        logger.info(f"保存检查点，当前手牌数: {self.total_hands}")
                        self.save_checkpoint(is_best=False)

                # ===== PPO训练提交 =====
                if self.model_type == 'ppo' and ppo_trajectories:
                    if async_trainer is not None:
                        # 异步模式：提交轨迹到写入缓冲
                        async_trainer.submit_trajectories(ppo_trajectories)
                        # 每隔一定批次交换缓冲并触发训练
                        if batch_counter % max(1, self.train_sync_every_batches) == 0:
                            async_trainer.swap_buffers()
                    else:
                        # 同步模式
                        _ppo_t0 = datetime.now().timestamp()
                        if hasattr(self.main_agent, 'append_trajectories_batch'):
                            self.main_agent.append_trajectories_batch(ppo_trajectories)
                        else:
                            for traj in ppo_trajectories:
                                self._apply_trajectory(traj)
                        _ppo_t1 = datetime.now().timestamp()
                        t_train_total += (_ppo_t1 - _ppo_t0)

                hand_idx += len(results_and_trajectories)
                batch_counter += 1

                # 每隔一段时间输出耗时分布
                if batch_counter % 10 == 0 and (t_env_total + t_train_total) > 0:
                    total_t = t_env_total + t_train_total
                    async_info = ""
                    if async_trainer is not None:
                        async_info = f", 异步训练累计{async_trainer._total_train_count}条/{async_trainer._total_train_time:.2f}s"
                    logger.info(
                        f"[GPU流水线] 耗时分布: 环境采样={t_env_total:.2f}s({t_env_total/total_t*100:.1f}%), "
                        f"同步训练={t_train_total:.2f}s({t_train_total/total_t*100:.1f}%), "
                        f"总耗时={total_t:.2f}s{async_info}"
                    )
        finally:
            # 停止异步训练线程
            if async_trainer is not None:
                async_trainer.stop()
                # 刷新剩余数据
                try:
                    self.main_agent.flush_pending_updates()
                except Exception as e:
                    err_msg = str(e).lower()
                    if "dnn library" in err_msg or "graph execution" in err_msg:
                        logger.warning("收尾PPO更新跳过(DNN/Graph错误): %s", e)
                    else:
                        raise
            if executor is not None:
                executor.shutdown(wait=True)

        logger.info(f"训练完成！总手牌: {self.total_hands}")
        if async_trainer is None and hasattr(self.main_agent, 'flush_pending_updates'):
            try:
                self.main_agent.flush_pending_updates()
            except Exception as e:
                err_msg = str(e).lower()
                if "dnn library" in err_msg or "graph execution" in err_msg or "xla_compile" in err_msg:
                    logger.warning(
                        "收尾时 PPO 剩余样本更新失败（多为 XLA/DNN 与当前 GPU 兼容问题），"
                        "检查点已保存，可忽略。若需避免可加 --disable-xla 重跑。错误: %s",
                        e,
                    )
                else:
                    raise
        if (
            (not self.tuning_mode)
            and self._last_regular_checkpoint_hands != self.total_hands
        ):
            self.save_checkpoint(is_best=False)
        if (not self.tuning_mode) and (not self.skip_final_eval):
            te0 = datetime.now().timestamp()
            final_eval = self.evaluate()
            te1 = datetime.now().timestamp()
            t_train_total += (te1 - te0)
            logger.info(f"最终评估: {final_eval}")
        else:
            logger.info("已按配置跳过最终评估（--skip-final-eval）")
        if num_hands > 0:
            logger.info(
                f"性能统计: 环境与决策总耗时={t_env_total:.2f}s, 同步训练总耗时={t_train_total:.2f}s, "
                f"平均每手环境耗时={t_env_total/num_hands:.4f}s"
            )

    def _gpu_pipeline_step(
        self,
        executor: ThreadPoolExecutor,
        remaining_hands: int,
        worker_state_reps: List,
        worker_opp_copies: List,
        gpu_bridge: GPUInferenceBridge,
    ) -> List[Tuple[Dict, Dict]]:
        """GPU流水线模式的一次训练步：
        1. 多线程并行推进环境，收集所有Main_AI待决策状态
        2. 主线程GPU批量推理
        3. 推理结果分发回各环境继续推进
        4. 循环直到所有手牌完成
        """
        worker_count = min(self.train_workers, remaining_hands)
        target_batch_hands = min(
            remaining_hands,
            worker_count * max(1, self.train_rollouts_per_worker)
        )

        # 创建游戏环境
        envs = []
        for _ in range(target_batch_hands):
            game, players = self._create_game()
            game.start_new_hand()
            envs.append({
                'game': game,
                'players': players,
                'last_step_info': None,
                'initial_stack': players[0].stack,
                'action_records': {p.name: [] for p in players},
                'all_in_context': {'happened': False, 'round': None, 'equity': 0.0},
                'trajectory': {'states': [], 'actions': [], 'old_probs': []},
                'done': False,
            })

        outputs: List[Tuple[Dict, Dict]] = []
        _outputs_lock = threading.Lock()  # 多牌桌并行时保护outputs
        remaining = len(envs)
        gpu_infer_count = 0

        def finalize_env(env_idx: int) -> None:
            """结算单个环境（线程安全）"""
            env = envs[env_idx]
            if env['done']:
                return
            game = env['game']
            players = env['players']
            alive = [p for p in players if p.is_active]
            if len(alive) == 1 and game.pot > 0:
                game._award_pot(alive[0])
            final_profit = players[0].stack - env['initial_stack']
            winners = [p.name for p in getattr(game, 'winners', [])]
            if not winners and hasattr(game, 'winner') and game.winner:
                winners = [game.winner.name]
            is_winner = "Main_AI" in winners

            final_reward = self.reward_calculator.calculate_reward(
                player=players[0], game=game, action="end",
                action_amount=0, is_winner=is_winner,
                profit_loss=final_profit, done=True,
            )
            main_actions = env['action_records'].get("Main_AI", [])
            if len(main_actions) == 1 and main_actions[0][0] == "fold" and final_profit <= 0:
                final_reward -= 0.25
            allin_unlucky_bonus = 0.0
            all_in_context = env['all_in_context']
            if (not is_winner) and all_in_context['happened'] and all_in_context['round'] in (1, 2):
                eq = all_in_context['equity']
                if eq >= self.allin_unlucky_eq_threshold:
                    unlucky_bonus = min(
                        self.allin_unlucky_bonus_cap,
                        0.20 + (eq - self.allin_unlucky_eq_threshold) * 2.0,
                    )
                    final_reward += unlucky_bonus
                    allin_unlucky_bonus = unlucky_bonus

            env['trajectory']['final_reward'] = final_reward
            env['trajectory']['allin_unlucky'] = {
                'triggered': allin_unlucky_bonus > 0,
                'bonus': allin_unlucky_bonus,
                'equity': all_in_context['equity'],
            }
            result = {
                'winner': game.winner.name if hasattr(game, 'winner') and game.winner else 'Unknown',
                'winners': winners,
                'player_profits': {p.name: p.stack - self.initial_stack for p in players},
                'action_histories': env['action_records'],
                'metrics': self._count_episode_metrics(
                    players[0], game,
                    env['action_records'].get("Main_AI", []),
                    lightweight=False,
                ),
            }
            with _outputs_lock:
                outputs.append((result, env['trajectory']))
            env['done'] = True

        def step_env_until_main_action(env_idx: int) -> Optional[np.ndarray]:
            """推进单个环境直到Main_AI需要决策，返回状态向量或None(结束)。"""
            env = envs[env_idx]
            if env['done']:
                return None
            game = env['game']
            players = env['players']

            while not game.is_hand_over():
                if game.is_round_over():
                    if not game.proceed_to_next_round():
                        finalize_env(env_idx)
                        return None
                    continue

                current_player = game.next_player()
                if not current_player or not current_player.is_active:
                    continue

                if current_player.name == "Main_AI":
                    # 需要GPU推理，返回状态
                    state_rep = worker_state_reps[env_idx % len(worker_state_reps)]
                    current_state = state_rep.get_state(game, current_player)
                    env['current_main_player'] = current_player
                    env['current_main_state'] = current_state
                    return current_state
                else:
                    # 对手动作：在worker本地处理
                    opp_agents = worker_opp_copies[env_idx % len(worker_opp_copies)]
                    try:
                        opp_idx = int(current_player.name.split('_')[-1]) - 1
                        if 0 <= opp_idx < len(opp_agents):
                            agent = opp_agents[opp_idx]
                        else:
                            agent = opp_agents[0]
                    except Exception:
                        agent = opp_agents[0]
                    action, amount = agent.select_action(game, current_player)
                    env['action_records'][current_player.name].append(
                        (action, amount if amount else 0)
                    )
                    if not game.handle_action(current_player, action, amount):
                        game.handle_action(current_player, 'fold', None)

            # 手牌结束
            finalize_env(env_idx)
            return None

        # 迭代式推进：收集所有Main_AI待决策状态 → GPU批量推理 → 分发结果
        max_iterations = 100  # 安全上限
        iteration = 0
        while remaining > 0 and iteration < max_iterations:
            iteration += 1
            # 1. 并行推进所有环境，收集待决策状态
            pending_states = []
            pending_env_indices = []

            active_indices = [i for i in range(len(envs)) if not envs[i]['done']]
            if len(active_indices) > 1 and executor is not None:
                # 多牌桌并行：用线程池并行推进所有活跃环境
                step_results = list(executor.map(step_env_until_main_action, active_indices))
                for env_idx, state in zip(active_indices, step_results):
                    if state is not None:
                        pending_states.append(state)
                        pending_env_indices.append(env_idx)
            else:
                for env_idx in active_indices:
                    state = step_env_until_main_action(env_idx)
                    if state is not None:
                        pending_states.append(state)
                        pending_env_indices.append(env_idx)

            # 2. GPU批量推理
            if pending_states:
                t_gpu0 = datetime.now().timestamp()
                states_batch = np.asarray(pending_states, dtype=np.float32)
                action_probs_batch, values_batch = gpu_bridge.batch_predict(states_batch)
                t_gpu1 = datetime.now().timestamp()
                gpu_infer_count += len(pending_states)

                # 3. 分发推理结果
                for i, env_idx in enumerate(pending_env_indices):
                    env = envs[env_idx]
                    if env['done'] or 'current_main_player' not in env:
                        continue
                    game = env['game']
                    current_player = env.pop('current_main_player')
                    current_state = env.pop('current_main_state')
                    action_probs = action_probs_batch[i]
                    value = values_batch[i]

                    # 应用合法动作掩码
                    legal_mask = _get_legal_action_mask(game, current_player)
                    masked_probs = action_probs * legal_mask
                    prob_sum = float(np.sum(masked_probs))
                    if prob_sum <= 1e-8:
                        masked_probs = legal_mask / (np.sum(legal_mask) + 1e-8)
                    else:
                        masked_probs = masked_probs / prob_sum
                    # 翻牌前位置感知过滤：压低弱牌在前位的加注概率
                    state_rep = worker_state_reps[env_idx % len(worker_state_reps)]
                    hand_strength = state_rep.calculate_hand_strength(current_player, game)
                    masked_probs = _apply_preflop_position_filter(masked_probs, game, current_player, hand_strength)
                    action_idx = int(np.random.choice(len(masked_probs), p=masked_probs))
                    action_vector = np.zeros(len(action_probs), dtype=np.float32)
                    action_vector[action_idx] = 1.0
                    action_str, amount = state_rep.decode_action(action_vector, game, current_player, hand_strength)

                    env['action_records'][current_player.name].append(
                        (action_str, amount if amount else 0)
                    )
                    success = game.handle_action(current_player, action_str, amount)
                    if success and (not env['all_in_context']['happened']) and current_player.is_all_in and game.round in (1, 2):
                        try:
                            eq = float(state_rep.calculate_hand_strength(current_player, game))
                        except Exception:
                            eq = 0.0
                        env['all_in_context'].update({
                            'happened': True,
                            'round': int(game.round),
                            'equity': max(0.0, min(1.0, eq)),
                        })

                    if not success:
                        game.handle_action(current_player, 'fold', None)
                        fold_prob = float(np.clip(masked_probs[0], 1e-8, 1.0))
                        env['trajectory']['states'].append(current_state)
                        env['trajectory']['actions'].append(0)
                        env['trajectory']['old_probs'].append(fold_prob)
                    else:
                        old_prob = float(np.clip(masked_probs[action_idx], 1e-8, 1.0))
                        env['trajectory']['states'].append(current_state)
                        env['trajectory']['actions'].append(int(action_idx))
                        env['trajectory']['old_probs'].append(old_prob)
                    env['last_step_info'] = (current_state, action_idx)

            # 更新剩余数
            remaining = sum(1 for e in envs if not e['done'])
            if not pending_states and remaining > 0:
                # 无进展时强制结算
                for env_idx in range(len(envs)):
                    if not envs[env_idx]['done']:
                        finalize_env(env_idx)
                break

        if gpu_infer_count > 0:
            logger.debug(
                f"[GPU流水线] 本批次{len(envs)}个并发牌桌, "
                f"GPU推理{gpu_infer_count}次, 完成{len(outputs)}手牌"
            )

        return outputs

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