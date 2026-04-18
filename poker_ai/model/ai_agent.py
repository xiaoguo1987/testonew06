# 德州扑克AI代理接口

import numpy as np
import logging
import threading
from typing import Tuple, Optional, Dict
from .state_representation import StateRepresentation
from .dqn_agent import DQNAgent, PolicyGradientAgent
from .ppo_model import PPOModel  # 添加PPO模型导入
from ..game.poker_game import Player, PokerGame

# 设置日志
logger = logging.getLogger('PokerAIAgent')


def _get_legal_action_mask(game: PokerGame, player: Player) -> np.ndarray:
    """
    5 维动作合法性掩码:
    0 fold
    1 call
    2 raise x2
    3 raise x4
    4 all-in
    """
    mask = np.zeros(5, dtype=np.float32)
    if not player.is_active or player.is_all_in:
        mask[0] = 1.0
        return mask

    call_amount = max(0, game.current_bet - player.bet)
    # fold / call 基本总是可用（call_amount=0 时等价于 check）
    mask[0] = 1.0
    if call_amount <= player.stack:
        mask[1] = 1.0

    current_bet = max(game.current_bet, game.big_blind)
    raise2_total = int(current_bet * 2)
    raise4_total = int(current_bet * 4)
    all_in_total = int(player.bet + player.stack)
    min_raise_total = max(game.current_bet * 2, game.big_blind)

    if raise2_total > game.current_bet and raise2_total >= min_raise_total and (raise2_total - player.bet) <= player.stack:
        mask[2] = 1.0
    if raise4_total > game.current_bet and raise4_total >= min_raise_total and (raise4_total - player.bet) <= player.stack:
        mask[3] = 1.0
    if all_in_total > game.current_bet and all_in_total >= min_raise_total and player.stack > 0:
        mask[4] = 1.0

    if np.sum(mask) <= 0:
        # 保底：至少允许 fold/call
        mask[0] = 1.0
        if call_amount <= player.stack:
            mask[1] = 1.0
    return mask

class GPUInferenceBridge:
    """GPU批量推理桥：在主线程统一调用GPU做批量推理，避免多线程GPU竞争。
    
    设计目标：
    - 采样worker不再持有完整模型副本，只持有StateRepresentation用于状态编码
    - 所有Main_AI的决策推理统一在主线程GPU上批量执行
    - 训练完成后自动同步权重快照给推理桥
    """

    def __init__(self, ppo_model: PPOModel):
        self._model = ppo_model
        self._lock = threading.Lock()
        # 推理用快照权重（训练线程更新后同步过来）
        self._snapshot_weights = None

    def batch_predict(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """在GPU上批量推理，返回 (action_probs, values)。
        
        使用训练模型的 predict_batch 方法，所有推理在主线程的GPU上执行。
        如果有未应用的权重快照，先应用再推理。
        """
        with self._lock:
            if self._snapshot_weights is not None:
                self._model.apply_weights_snapshot(self._snapshot_weights)
                self._snapshot_weights = None
        return self._model.predict_batch(states)

    def sync_weights_from_main(self, snapshot):
        """训练完成后，异步训练线程调用此方法更新推理权重。"""
        with self._lock:
            self._snapshot_weights = snapshot


class LightweightInferenceAgent:
    """轻量级推理Agent：只持有StateRepresentation，不做模型推理。
    
    用于采样worker：worker只需编码状态，不需要做模型推理。
    推理统一由主线程的GPUInferenceBridge完成。
    """

    def __init__(self, state_rep: StateRepresentation, agent_type: str = 'ppo'):
        self.state_rep = state_rep
        self.agent_type = agent_type
        self.name = f"LightweightInference_{agent_type}"

    def encode_state(self, game, player) -> np.ndarray:
        """编码游戏状态为向量"""
        return self.state_rep.get_state(game, player)


class PokerAIAgent:
    """德州扑克AI代理统一接口"""
    
    def __init__(self, agent_type: str = 'dqn', state_dim: Optional[int] = None, target_update_freq: int = 10,
                 learning_rate: float = 0.001, optimizer: str = 'adam',
                 ppo_clip_epsilon: Optional[float] = None, ppo_update_epochs: Optional[int] = None,
                 ppo_entropy_coef: Optional[float] = None, ppo_batch_size: Optional[int] = None,
                 ppo_update_min_steps: Optional[int] = None):
        """
        初始化AI代理
        agent_type: 'dqn'、'policy_gradient' 或 'ppo'
        state_dim: 状态维度，如果为None则自动计算
        target_update_freq: DQN目标网络更新频率（仅DQN代理有效）
        learning_rate: 学习率
        optimizer: 优化器类型
        ppo_clip_epsilon: PPO clip 参数（仅 PPO）
        ppo_update_epochs: PPO 每批更新轮次（仅 PPO）
        """
        self.state_rep = StateRepresentation()
        if state_dim is None:
            state_dim = self.state_rep.total_state_dim
        # 保存初始化参数，供 clone_for_inference 复用，避免副本超参数退化到硬编码默认值
        self._init_kwargs = dict(
            agent_type=agent_type,
            state_dim=state_dim,
            target_update_freq=target_update_freq,
            learning_rate=learning_rate,
            optimizer=optimizer,
            ppo_clip_epsilon=ppo_clip_epsilon,
            ppo_update_epochs=ppo_update_epochs,
            ppo_entropy_coef=ppo_entropy_coef,
            ppo_batch_size=ppo_batch_size,
            ppo_update_min_steps=ppo_update_min_steps,
        )

        if agent_type == 'dqn':
            self.agent = DQNAgent(state_dim=state_dim, target_update_freq=target_update_freq,
                                 learning_rate=learning_rate, optimizer=optimizer)
        elif agent_type == 'policy_gradient':
            self.agent = PolicyGradientAgent(state_dim=state_dim)
        elif agent_type == 'ppo':
            # 构建PPOConfig，确保所有优化参数（KL早停、value clip等）正确传递
            from ..config.ppo_config import PPOConfig
            cfg_kwargs = dict(input_dim=state_dim, lr=learning_rate)
            if ppo_clip_epsilon is not None:
                cfg_kwargs['clip_epsilon'] = ppo_clip_epsilon
            if ppo_update_epochs is not None:
                cfg_kwargs['update_epochs'] = ppo_update_epochs
            if ppo_entropy_coef is not None:
                cfg_kwargs['entropy_coef'] = ppo_entropy_coef
            if ppo_batch_size is not None:
                cfg_kwargs['batch_size'] = ppo_batch_size
            ppo_cfg = PPOConfig(**cfg_kwargs)
            self.agent = PPOModel(config=ppo_cfg)
            auto_min_steps = max(256, int(self.agent.batch_size) * 8)
            self.ppo_update_min_steps = int(ppo_update_min_steps) if ppo_update_min_steps else auto_min_steps
            self.ppo_pending_buffer = {
                'states': [], 'actions': [], 'rewards': [], 'dones': [], 'old_probs': []
            }
        else:
            raise ValueError(f"不支持的代理类型: {agent_type}")
        
        self.agent_type = agent_type
        self.name = f"AI_Agent_{agent_type}"
    
    def select_action(self, game: PokerGame, player: Player) -> Tuple[str, Optional[int]]:
        """
        根据当前游戏状态选择动作
        返回: (动作类型, 动作金额)
        """
        action, amount, _ = self.select_action_with_info(game, player)
        return action, amount

    def select_action_with_info(self, game: PokerGame, player: Player) -> Tuple[str, Optional[int], Dict]:
        """
        选择动作并返回训练所需的额外信息。
        PPO 场景下可避免重复前向推理，并确保记录的是“实际执行动作”。
        """
        state = self.state_rep.get_state(game, player)
        if self.agent_type == 'ppo':
            action_probs, value = self.agent.predict(state)
            legal_mask = _get_legal_action_mask(game, player)
            masked_probs = action_probs * legal_mask
            prob_sum = float(np.sum(masked_probs))
            if prob_sum <= 1e-8:
                masked_probs = legal_mask / (np.sum(legal_mask) + 1e-8)
            else:
                masked_probs = masked_probs / prob_sum
            action_idx = int(np.random.choice(len(masked_probs), p=masked_probs))
            hand_strength = self.state_rep.calculate_hand_strength(player, game)
            action_vector = np.zeros(len(action_probs))
            action_vector[action_idx] = 1
            action, amount = self.state_rep.decode_action(action_vector, game, player, hand_strength)
            info = {
                'state': state,
                'action_idx': action_idx,
                'old_prob': float(np.clip(masked_probs[action_idx], 1e-8, 1.0)),
                'action_probabilities': masked_probs,
                'state_value': float(value),
            }
            return action, amount, info

        action, amount = self.agent.act(state, game, player)
        return action, amount, {'state': state}

    def select_actions_batch_with_info(
        self,
        game_player_pairs,
        precomputed_states: Optional[np.ndarray] = None,
    ):
        """
        批量选择动作。
        - PPO: 单次前向推理多局状态
        - 其他模型: 退化为逐条调用，保持兼容
        """
        if not game_player_pairs:
            return []

        if self.agent_type != 'ppo':
            results = []
            for idx, (game, player) in enumerate(game_player_pairs):
                state = None if precomputed_states is None else precomputed_states[idx]
                if state is None:
                    action, amount, info = self.select_action_with_info(game, player)
                else:
                    action, amount, info = self._select_action_with_state(game, player, state)
                results.append((action, amount, info))
            return results

        if precomputed_states is None:
            states = np.asarray(
                [self.state_rep.get_state(game, player) for game, player in game_player_pairs],
                dtype=np.float32,
            )
        else:
            states = np.asarray(precomputed_states, dtype=np.float32)

        action_probs_batch, values_batch = self.agent.predict_batch(states)
        results = []
        for idx, ((game, player), action_probs, value) in enumerate(
            zip(game_player_pairs, action_probs_batch, values_batch)
        ):
            legal_mask = _get_legal_action_mask(game, player)
            masked_probs = action_probs * legal_mask
            prob_sum = float(np.sum(masked_probs))
            if prob_sum <= 1e-8:
                masked_probs = legal_mask / (np.sum(legal_mask) + 1e-8)
            else:
                masked_probs = masked_probs / prob_sum
            action_idx = int(np.random.choice(len(masked_probs), p=masked_probs))
            hand_strength = self.state_rep.calculate_hand_strength(player, game)
            action_vector = np.zeros(len(action_probs), dtype=np.float32)
            action_vector[action_idx] = 1.0
            action, amount = self.state_rep.decode_action(action_vector, game, player, hand_strength)
            info = {
                'state': states[idx],
                'action_idx': action_idx,
                'old_prob': float(np.clip(masked_probs[action_idx], 1e-8, 1.0)),
                'action_probabilities': masked_probs,
                'state_value': float(value),
            }
            results.append((action, amount, info))
        return results

    def _select_action_with_state(
        self,
        game: PokerGame,
        player: Player,
        state: np.ndarray,
    ) -> Tuple[str, Optional[int], Dict]:
        """给批量路径复用的单状态版本，避免重复编码 state。"""
        if self.agent_type == 'ppo':
            action_probs, value = self.agent.predict(state)
            legal_mask = _get_legal_action_mask(game, player)
            masked_probs = action_probs * legal_mask
            prob_sum = float(np.sum(masked_probs))
            if prob_sum <= 1e-8:
                masked_probs = legal_mask / (np.sum(legal_mask) + 1e-8)
            else:
                masked_probs = masked_probs / prob_sum
            action_idx = int(np.random.choice(len(masked_probs), p=masked_probs))
            hand_strength = self.state_rep.calculate_hand_strength(player, game)
            action_vector = np.zeros(len(action_probs), dtype=np.float32)
            action_vector[action_idx] = 1.0
            action, amount = self.state_rep.decode_action(action_vector, game, player, hand_strength)
            return action, amount, {
                'state': state,
                'action_idx': action_idx,
                'old_prob': float(np.clip(masked_probs[action_idx], 1e-8, 1.0)),
                'action_probabilities': masked_probs,
                'state_value': float(value),
            }

        action, amount = self.agent.act(state, game, player)
        return action, amount, {'state': state}
    
    def learn(self, reward: float, state: Optional[np.ndarray] = None, 
              action: Optional[int] = None, next_state: Optional[np.ndarray] = None, 
              done: Optional[bool] = False) -> None:
        """
        从经验中学习
        """
        if self.agent_type == 'dqn' and all(v is not None for v in [state, action, next_state, done]):
            # DQN代理需要完整的经验元组
            self.agent.remember(state, action, reward, next_state, done)
            self.agent.replay()
        elif self.agent_type == 'policy_gradient':
            # 策略梯度代理只需要记住奖励
            self.agent.remember_reward(reward)
            # 回合结束时训练
            if done:
                self.agent.train()
        elif self.agent_type == 'ppo':
            # PPO 使用轨迹级学习：由 append_step + finish_trajectory 驱动，此处仅兼容旧式单步 learn
            if all(v is not None for v in [state, action, next_state, done]):
                self._ppo_ensure_buffer()
                old_probs, _ = self.agent.predict(state)
                old_prob = float(np.clip(old_probs[action], 1e-8, 1.0))
                self.ppo_buffer['states'].append(state)
                self.ppo_buffer['actions'].append(action)
                self.ppo_buffer['rewards'].append(reward)
                self.ppo_buffer['dones'].append(done)
                self.ppo_buffer['old_probs'].append(old_prob)
                if done and len(self.ppo_buffer['states']) > 0:
                    self._ppo_train_from_buffer()
                    self._ppo_clear_buffer()

    def learn_from_transitions(self, transitions, replay_updates: int = 1) -> None:
        """
        批量应用转移样本，降低 Python 调度开销。
        - DQN: 先批量 remember，再 replay 指定次数
        - 其他模型: 回退到逐条 learn
        """
        if not transitions:
            return
        if self.agent_type == 'dqn':
            for (state, action, reward, next_state, done) in transitions:
                self.agent.remember(state, action, reward, next_state, done)
            for _ in range(max(1, int(replay_updates))):
                self.agent.replay()
            return

        for (state, action, reward, next_state, done) in transitions:
            self.learn(
                reward=reward,
                state=state,
                action=action,
                next_state=next_state,
                done=done,
            )

    def _ppo_ensure_buffer(self) -> None:
        if not hasattr(self, 'ppo_buffer'):
            self.ppo_buffer = {
                'states': [], 'actions': [], 'rewards': [], 'dones': [], 'old_probs': []
            }

    def _ppo_clear_buffer(self) -> None:
        self.ppo_buffer = {
            'states': [], 'actions': [], 'rewards': [], 'dones': [], 'old_probs': []
        }

    def _ppo_train_from_buffer(self) -> None:
        buf = self.ppo_buffer
        states = np.array(buf['states'])
        actions = np.array(buf['actions'])
        rewards = np.array(buf['rewards'], dtype=np.float32)
        dones = np.array(buf['dones'])
        old_probs = np.array(buf['old_probs'], dtype=np.float32)
        try:
            self.agent.train(states, actions, rewards, dones, old_probs)
        except Exception as e:
            err_msg = str(e).lower()
            if "dnn library" in err_msg or "graph execution" in err_msg:
                logger.warning(
                    "本次 PPO 更新因 DNN/Graph 错误跳过（可尝试加 --disable-xla 与 --disable-onednn）: %s",
                    e,
                )
            else:
                raise

    def _ppo_train_pending_buffer(self, force: bool = False) -> None:
        """仅 PPO：当累计样本达到阈值（或 force=True）时，批量训练一次。"""
        if self.agent_type != 'ppo':
            return
        pending = getattr(self, 'ppo_pending_buffer', None)
        if not pending or len(pending['states']) == 0:
            return
        if (not force) and len(pending['states']) < self.ppo_update_min_steps:
            return

        import time as _time
        _t0 = _time.time()
        n_samples = len(pending['states'])
        logger.info(
            f"[GPU诊断] 触发PPO训练: 累计样本={n_samples}, "
            f"阈值={self.ppo_update_min_steps}, force={force}"
        )

        states = np.array(pending['states'])
        actions = np.array(pending['actions'])
        rewards = np.array(pending['rewards'], dtype=np.float32)
        dones = np.array(pending['dones'])
        old_probs = np.array(pending['old_probs'], dtype=np.float32)
        try:
            self.agent.train(states, actions, rewards, dones, old_probs)
        except Exception as e:
            err_msg = str(e).lower()
            if "dnn library" in err_msg or "graph execution" in err_msg:
                logger.warning(
                    "本次 PPO 更新因 DNN/Graph 错误跳过（可尝试加 --disable-xla 与 --disable-onednn 重跑）: %s",
                    e,
                )
            else:
                raise
        _elapsed = _time.time() - _t0
        logger.info(
            f"[GPU诊断] PPO训练完成: 样本数={n_samples}, 耗时={_elapsed:.4f}s, "
            f"吞吐={n_samples/_elapsed:.0f} samples/s"
        )
        self.ppo_pending_buffer = {
            'states': [], 'actions': [], 'rewards': [], 'dones': [], 'old_probs': []
        }

    def append_trajectories_batch(self, trajectories) -> None:
        """
        仅 PPO：批量写入多条轨迹后再触发一次阈值检查，减少每手牌一次函数调用的开销。
        trajectories: [{'states','actions','old_probs','final_reward'}, ...]
        """
        if self.agent_type != 'ppo' or not trajectories:
            return
        pending = self.ppo_pending_buffer
        for traj in trajectories:
            states = traj.get('states', [])
            actions = traj.get('actions', [])
            old_probs = traj.get('old_probs', [])
            n = len(states)
            if n == 0:
                continue
            final_reward = float(traj.get('final_reward', 0.0))
            rewards = [0.0] * (n - 1) + [final_reward]
            dones = [False] * (n - 1) + [True]
            pending['states'].extend(states)
            pending['actions'].extend(actions)
            pending['old_probs'].extend(old_probs)
            pending['rewards'].extend(rewards)
            pending['dones'].extend(dones)
        self._ppo_train_pending_buffer(force=False)

    def append_step(self, state: np.ndarray, action: int, old_prob: float) -> None:
        """仅 PPO：每步将 (state, action, old_prob) 加入轨迹，不传 reward。"""
        if self.agent_type != 'ppo':
            return
        self._ppo_ensure_buffer()
        self.ppo_buffer['states'].append(state)
        self.ppo_buffer['actions'].append(action)
        self.ppo_buffer['old_probs'].append(float(np.clip(old_prob, 1e-8, 1.0)))

    def finish_trajectory(self, final_reward: float) -> None:
        """仅 PPO：一手牌结束时写入累计缓冲，并在达到阈值时批量 PPO 更新。"""
        if self.agent_type != 'ppo':
            return
        self._ppo_ensure_buffer()
        n = len(self.ppo_buffer['states'])
        if n == 0:
            return
        self.ppo_buffer['rewards'] = [0.0] * (n - 1) + [final_reward]
        self.ppo_buffer['dones'] = [False] * (n - 1) + [True]
        pending = self.ppo_pending_buffer
        pending['states'].extend(self.ppo_buffer['states'])
        pending['actions'].extend(self.ppo_buffer['actions'])
        pending['rewards'].extend(self.ppo_buffer['rewards'])
        pending['dones'].extend(self.ppo_buffer['dones'])
        pending['old_probs'].extend(self.ppo_buffer['old_probs'])
        self._ppo_clear_buffer()
        self._ppo_train_pending_buffer(force=False)

    def flush_pending_updates(self) -> None:
        """训练结束/评估前可调用，确保 PPO 剩余样本也被训练。"""
        if self.agent_type == 'ppo':
            self._ppo_train_pending_buffer(force=True)
    
    def get_decision_info(self, game: PokerGame, player: Player) -> Dict:
        """
        获取决策信息（用于提供建议）
        返回: 包含决策依据的字典
        """
        state = self.state_rep.get_state(game, player)
        decision_info = {}
        
        if self.agent_type == 'dqn':
            # 获取Q值
            q_values = self.agent.get_q_values(state)
            decision_info['q_values'] = q_values.tolist()
            decision_info['best_action_index'] = int(np.argmax(q_values))
        elif self.agent_type == 'policy_gradient':
            # 获取动作概率
            action_probs = self.agent.get_action_probs(state)
            decision_info['action_probabilities'] = action_probs.tolist()
            decision_info['most_likely_action'] = int(np.argmax(action_probs))
        elif self.agent_type == 'ppo':
            # 获取动作概率和状态价值
            action_probs, value = self.agent.predict(state)
            decision_info['action_probabilities'] = action_probs.tolist()
            decision_info['state_value'] = float(value)
            decision_info['most_likely_action'] = int(np.argmax(action_probs))
        
        # 添加游戏状态信息
        decision_info['current_bet'] = game.current_bet
        decision_info['player_bet'] = player.bet
        decision_info['pot_size'] = game.pot
        decision_info['round'] = game.round
        
        return decision_info
    
    def load_model(self, filepath: str) -> bool:
        """加载模型权重"""
        try:
            # 必须调用 self.agent.load，而不是 self.load_model
            self.agent.load(filepath) 
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def save_model(self, filepath: str) -> None:
        """保存模型权重"""
        self.agent.save(filepath)
    
    # === 新增：兼容 CheckpointManager 的别名方法 ===
    def save(self, filepath: str) -> None:
        """兼容性接口：调用 save_model"""
        self.save_model(filepath)

    def load(self, filepath: str) -> None:
        """兼容性接口：调用 load_model"""
        self.load_model(filepath)
    # ============================================

    def get_weights(self):
        """获取代理权重"""
        if hasattr(self.agent, 'get_weights'):
            return self.agent.get_weights()
        return None

    def set_weights(self, weights):
        """设置代理权重"""
        if hasattr(self.agent, 'set_weights'):
            self.agent.set_weights(weights)

    def clone_for_inference(self) -> "PokerAIAgent":
        """创建仅用于推理的副本（不包含 replay/optimizer 等训练状态），供并行采样 worker 使用。"""
        kwargs = dict(self._init_kwargs)
        if self.agent_type == 'ppo':
            kwargs['ppo_clip_epsilon'] = getattr(
                self.agent, 'clip_epsilon', kwargs.get('ppo_clip_epsilon')
            )
            kwargs['ppo_update_epochs'] = getattr(
                self.agent, 'update_epochs', kwargs.get('ppo_update_epochs')
            )
            kwargs['ppo_entropy_coef'] = getattr(
                self.agent, 'entropy_coef', kwargs.get('ppo_entropy_coef')
            )
            kwargs['ppo_batch_size'] = getattr(
                self.agent, 'batch_size', kwargs.get('ppo_batch_size')
            )
            kwargs['ppo_update_min_steps'] = getattr(
                self, 'ppo_update_min_steps', kwargs.get('ppo_update_min_steps')
            )
        clone = PokerAIAgent(**kwargs)
        weights = self.get_weights()
        if weights is not None:
            clone.set_weights(weights)
        if self.agent_type == 'dqn' and hasattr(self.agent, 'epsilon'):
            clone.agent.epsilon = self.agent.epsilon
            clone.agent.epsilon_min = getattr(self.agent, 'epsilon_min', 0.1)
            clone.agent.epsilon_decay = getattr(self.agent, 'epsilon_decay', 0.995)
        return clone

    def update_target_model(self) -> None:
        """更新目标模型（DQN特定）"""
        if self.agent_type == 'dqn':
            self.agent.update_target_model()
    
    def get_epsilon(self) -> Optional[float]:
        """获取探索率（DQN特定）"""
        if self.agent_type == 'dqn':
            return self.agent.epsilon
        return None
    
    def set_epsilon(self, value: float, min_value: Optional[float] = None, decay_rate: Optional[float] = None) -> None:
        """设置探索参数（DQN特定）
        
        Args:
            value: 当前探索率值
            min_value: 最小探索率值（可选）
            decay_rate: 衰减率值（可选）
        """
        if self.agent_type == 'dqn' and hasattr(self.agent, 'epsilon'):
            # 设置当前探索率，确保在合理范围内
            self.agent.epsilon = max(0.01, min(1.0, value))
            
            # 设置最小探索率（如果提供）
            if min_value is not None and hasattr(self.agent, 'epsilon_min'):
                self.agent.epsilon_min = max(0.01, min(1.0, min_value))
            
            # 设置衰减率（如果提供）
            if decay_rate is not None and hasattr(self.agent, 'epsilon_decay'):
                self.agent.epsilon_decay = max(0.00001, min(0.1, decay_rate))
            
            # 日志记录
            logger.info(f"更新DQN探索参数 - epsilon: {self.agent.epsilon}, " 
                      f"epsilon_min: {getattr(self.agent, 'epsilon_min', 'N/A')}, "
                      f"epsilon_decay: {getattr(self.agent, 'epsilon_decay', 'N/A')}")

# 创建一个组合多个AI代理的集成模型
class EnsembleAgent:
    """集成多个AI代理的决策模型"""
    
    def __init__(self, agent_types: list = ['dqn', 'policy_gradient']):
        """
        初始化集成代理
        agent_types: 要集成的代理类型列表
        """
        state_rep = StateRepresentation()
        state_dim = state_rep.total_state_dim
        
        self.agents = []
        for agent_type in agent_types:
            self.agents.append(PokerAIAgent(agent_type=agent_type, state_dim=state_dim))
        
        self.state_rep = state_rep
    
    def select_action(self, game: PokerGame, player: Player) -> Tuple[str, Optional[int]]:
        """
        通过投票机制选择最终动作
        """
        # 收集所有代理的动作建议
        actions = []
        for agent in self.agents:
            action = agent.select_action(game, player)
            actions.append(action[0])  # 只考虑动作类型
        
        # 统计动作类型的投票数
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # 选择得票最多的动作类型
        best_action_type = max(action_counts.keys(), key=lambda x: action_counts[x])
        
        # 如果选中的是raise，使用第一个代理的金额建议
        if best_action_type == 'raise':
            for agent in self.agents:
                action = agent.select_action(game, player)
                if action[0] == 'raise' and action[1] is not None:
                    return (best_action_type, action[1])
        elif best_action_type == 'call':
            # 跟注金额固定
            call_amount = max(0, game.current_bet - player.bet)
            return (best_action_type, call_amount)
        
        return (best_action_type, None)
    
    def learn(self, reward: float, done: bool = False) -> None:
        """
        更新所有代理
        """
        for agent in self.agents:
            # 对于集成模型，简化学习过程
            agent.learn(reward, done=done)
    
    def get_decision_info(self, game: PokerGame, player: Player) -> Dict:
        """
        获取集成决策信息
        """
        info = {
            'ensemble_decisions': [],
            'consensus': {}
        }
        
        # 收集每个代理的决策信息
        for i, agent in enumerate(self.agents):
            agent_info = agent.get_decision_info(game, player)
            agent_info['agent_type'] = agent.agent_type
            info['ensemble_decisions'].append(agent_info)
        
        # 计算动作投票
        actions = []
        for agent in self.agents:
            action = agent.select_action(game, player)
            actions.append(action[0])
        
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        info['consensus']['action_votes'] = action_counts
        
        return info
    
    def save_models(self, base_path: str) -> None:
        """
        保存所有代理的模型
        """
        for i, agent in enumerate(self.agents):
            filepath = f"{base_path}_{agent.agent_type}_{i}.h5"
            agent.save_model(filepath)
    
    def load_models(self, base_path: str) -> None:
        """
        加载所有代理的模型
        """
        for i, agent in enumerate(self.agents):
            filepath = f"{base_path}_{agent.agent_type}_{i}.h5"
            try:
                agent.load_model(filepath)
            except Exception as e:
                print(f"无法加载模型 {filepath}: {e}")

# 示例使用
if __name__ == "__main__":
    from ..game.poker_game import Player, PokerGame
    
    # 创建玩家
    players = [
        Player("Human", 1000, False),
        Player("AI_1", 1000, True),
        Player("AI_2", 1000, True),
        Player("AI_3", 1000, True),
        Player("AI_4", 1000, True),
        Player("AI_5", 1000, True)
    ]
    
    # 创建游戏
    game = PokerGame(5, 10, players)
    game.start_new_hand()
    
    # 创建单个AI代理
    ai_agent = PokerAIAgent(agent_type='dqn')
    
    # 创建PPO代理
    ppo_agent = PokerAIAgent(agent_type='ppo')
    
    # 创建集成代理
    ensemble_agent = EnsembleAgent()
    
    # 测试动作选择
    ai_action = ai_agent.select_action(game, players[1])
    print(f"单个AI代理选择的动作: {ai_action}")
    
    ppo_action = ppo_agent.select_action(game, players[2])
    print(f"PPO代理选择的动作: {ppo_action}")
    
    ensemble_action = ensemble_agent.select_action(game, players[3])
    print(f"集成代理选择的动作: {ensemble_action}")
    
    # 测试决策信息
    decision_info = ai_agent.get_decision_info(game, players[1])
    print(f"\n决策信息示例: {decision_info.keys()}")
    
    if 'q_values' in decision_info:
        print(f"Q值: {decision_info['q_values']}")
    elif 'action_probabilities' in decision_info:
        print(f"动作概率: {decision_info['action_probabilities']}")