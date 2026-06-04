"""SEED RL GPU-Centric 训练器 — 集中式 GPU 推理 + PPO 训练

架构：
    Main Process (GPU)
    ├── inference dispatcher: 从 worker Pipes 收集状态 → GPU batch → 分发结果
    ├── trajectory collector: 从 traj_queue 收集轨迹
    └── PPO trainer: 积累足够样本后在 GPU 上训练

    Worker1..WorkerN (CPU, 独立进程)
    └── 纯游戏逻辑（PokerGame），通过 Pipe 请求 GPU 推理
"""

import os
import time
import logging
import json
import math
import threading
import shutil
import random
import glob
import subprocess
import numpy as np
import multiprocessing as mp
from multiprocessing import Queue, Process, Pipe
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

from ..action_space import ACTION_DIM, ACTION_NAMES

logger = logging.getLogger('SeedRLTrainer')


class SeedRLTrainer:
    """SEED RL 风格的 GPU-Centric 训练器。

    所有 NN 计算集中在主进程 GPU 上，CPU worker 只跑游戏逻辑。
    """

    def __init__(
        self,
        model,
        checkpoint_dir: str = './checkpoints',
        small_blind: int = 5,
        big_blind: int = 10,
        initial_stack: int = 1000,
        num_workers: int = 64,
        eval_interval: int = 200,
        checkpoint_interval: int = 1000,
        num_eval_games: int = 100,
        opponent_update_interval: int = 1000,
        inference_min_batch: int = 32,
        inference_max_batch: int = 512,
        inference_timeout_s: float = 0.002,  # 推理批量积攒超时（从1ms提升到2ms：给批量更多积攒时间，提高GPU吞吐）
        ppo_buffer_size: int = 4096,
        skip_final_eval: bool = False,
        skip_final_buffer_train: bool = False,
        allin_eq_threshold: float = 0.72,
        allin_bonus_cap: float = 0.25,
        tables_per_worker: int = 8,
        metrics_output: Optional[str] = None,
        eval_seed: Optional[int] = 20260530,
        curriculum_start_phase: int = 1,
        curriculum_max_phase: int = 3,
        opponent_score_file: Optional[str] = None,
        hard_opponent_temperature: float = 200.0,
        hard_opponent_negative_ratio: float = 0.0,
        hard_opponent_scored_only_ratio: float = 0.0,
        hard_opponent_positive_ratio: float = 0.0,
        opponent_ensemble_size: int = 1,
        opponent_roster_file: Optional[str] = None,
        checkpoint_validation_command: Optional[str] = None,
        stop_on_validation_failure: bool = False,
        validation_start_hands: int = 0,
        max_anchor_kl: float = 0.0,
        anchor_kl_patience: int = 10,
        max_reference_kl: float = 0.0,
        reference_kl_patience: int = 10,
    ):
        """
        Args:
            model: LargePPOModel 实例（已在 GPU 上构建）
            num_workers: CPU 游戏 worker 进程数
            tables_per_worker: 每个 worker 同时管理的牌桌数（向量化环境）
            inference_min_batch: 推理最小批量
            inference_max_batch: 推理最大批量
            inference_timeout_s: 推理收集超时（秒）
            ppo_buffer_size: 积累多少步触发一次 PPO 训练
        """
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.initial_stack = initial_stack
        self.num_workers = num_workers
        self.eval_interval = eval_interval
        self.checkpoint_interval = checkpoint_interval
        self.num_eval_games = num_eval_games
        self.opponent_update_interval = opponent_update_interval
        self.inference_min_batch = inference_min_batch
        self.inference_max_batch = inference_max_batch
        self.inference_timeout_s = inference_timeout_s
        self.ppo_buffer_size = ppo_buffer_size
        self.skip_final_eval = skip_final_eval
        self.skip_final_buffer_train = bool(skip_final_buffer_train)
        self.allin_eq_threshold = allin_eq_threshold
        self.allin_bonus_cap = allin_bonus_cap
        self.tables_per_worker = tables_per_worker
        self.metrics_output = metrics_output or os.path.join(checkpoint_dir, "metrics.jsonl")
        self.eval_seed = eval_seed
        self.hard_opponent_temperature = max(float(hard_opponent_temperature), 1e-6)
        self.hard_opponent_negative_ratio = min(max(float(hard_opponent_negative_ratio), 0.0), 1.0)
        self.hard_opponent_scored_only_ratio = min(max(float(hard_opponent_scored_only_ratio), 0.0), 1.0)
        self.hard_opponent_positive_ratio = min(max(float(hard_opponent_positive_ratio), 0.0), 1.0)
        self.opponent_ensemble_size = max(1, int(opponent_ensemble_size))
        self.opponent_roster_file = opponent_roster_file
        self.checkpoint_validation_command = checkpoint_validation_command
        self.stop_on_validation_failure = bool(stop_on_validation_failure)
        self.validation_start_hands = max(0, int(validation_start_hands))
        self.max_anchor_kl = float(max_anchor_kl) if float(max_anchor_kl) > 0.0 else None
        self.anchor_kl_patience = max(1, int(anchor_kl_patience))
        self._anchor_kl_high_streak = 0
        self.max_reference_kl = float(max_reference_kl) if float(max_reference_kl) > 0.0 else None
        self.reference_kl_patience = max(1, int(reference_kl_patience))
        self._reference_kl_high_streak = 0

        # 训练统计
        self.total_hands = 0
        self.total_ppo_updates = 0
        self.best_score = -float('inf')
        self.num_hands_target = 0  # 训练目标手牌数，在 train() 中设置
        self._curriculum_base_hands = 0  # resume 后按本轮新增手数计算 curriculum

        # Curriculum Learning 阶段定义
        # Phase 1 (0-20%): 全鱼对手 → 学习基本策略（别乱弃牌、基本下注）
        # Phase 2 (20-50%): 混合对手（3鱼+2旧版） → 过渡阶段
        # Phase 3 (50%+): 自对弈（3PPO+2鱼） → 防止策略坍缩
        self._curriculum_min_phase = max(0, min(2, int(curriculum_start_phase) - 1))
        self._curriculum_max_phase = max(self._curriculum_min_phase, min(2, int(curriculum_max_phase) - 1))
        self._curriculum_phase = self._curriculum_min_phase  # 当前阶段
        self._curriculum_thresholds = (0.20, 0.50)  # 阶段切换阈值
        self._curriculum_opp_configs = [
            ['fish', 'fish', 'fish', 'fish', 'fish'],      # Phase 1: 全鱼
            ['ppo', 'ppo', 'fish', 'fish', 'fish'],        # Phase 2: 混合
            ['ppo', 'ppo', 'ppo', 'fish', 'fish'],         # Phase 3: 自对弈（保留2鱼防坍缩）
        ]
        self.opp_type_list = self._curriculum_opp_configs[self._curriculum_phase]
        # Worker 侧对手类型固定为 PPO+Fish（始终发送推理请求）
        # Curriculum 通过主进程推理调度器实现：对鱼对手返回随机动作
        self._worker_opp_type_list = ['ppo', 'ppo', 'ppo', 'ppo', 'fish']

        # ====== 初始化专属的对手模型，彻底消除权重切换开销 ======
        from ..model.large_ppo_model import LargePPOModel
        self.opp_model = LargePPOModel(
            input_dim=model.input_dim,
            output_dim=model.output_dim,
            hidden_dims=model.hidden_dims,
            residual_layer_idx=model.residual_layer_idx,
        )
        # 初始化时就同步一次权重
        self.opp_model.set_weights(model.get_weights())
        self._opponent_models = [self.opp_model]
        for _ in range(1, self.opponent_ensemble_size):
            extra_opp = LargePPOModel(
                input_dim=model.input_dim,
                output_dim=model.output_dim,
                hidden_dims=model.hidden_dims,
                residual_layer_idx=model.residual_layer_idx,
            )
            extra_opp.set_weights(model.get_weights())
            self._opponent_models.append(extra_opp)
        self._opponent_ensemble_paths: List[str] = ["current"] * len(self._opponent_models)
        self._opp_weights = None  # 废弃旧的权重快照变量

        # 历史对手池 (Fictitious Self-Play)
        # 保存历史 checkpoint 路径，对手更新时随机抽取（近期权重更高）
        self._opponent_pool: List[str] = []  # 可用的历史 checkpoint 路径列表
        self._opponent_score_paths: List[str] = []
        self._opponent_scores: Dict[str, float] = self._load_opponent_scores(opponent_score_file)
        self._opponent_roster_weighted = False
        self._opponent_roster_buckets = self._load_opponent_roster(opponent_roster_file)
        self._opponent_pool_max_size = 20    # 对手池最大容量
        self._loaded_checkpoint_path: Optional[str] = None
        self._archive_dir = '/root/autodl-tmp/poker_checkpoints'  # 数据盘归档目录
        self._max_keep_local = 30  # 本地保留最近30个 checkpoint
        os.makedirs(self._archive_dir, exist_ok=True)

        self._latest_win_rate = 0.5  # 最近一次评估胜率，用于安全阀判断
        self._latest_eval_bb_per_100 = 0.0  # 德扑收益比单纯胜率更重要
        self._low_win_eval_streak = 0  # 连续低胜率评估次数，避免一次噪声评估触发长期降级

        # 推理与训练诊断计数器，写入 metrics.jsonl 供 agent 分析。
        self._inference_batches = 0
        self._inference_items = 0
        self._inference_main_items = 0
        self._inference_opp_items = 0
        self._inference_fish_items = 0
        self._inference_seconds = 0.0
        self._last_ppo_seconds = 0.0
        self._last_ppo_steps = 0
        self._traj_overflow = []
        self._action_names = ACTION_NAMES
        self._train_action_counts = np.zeros(ACTION_DIM, dtype=np.int64)
        self._last_metric_action_counts = np.zeros(ACTION_DIM, dtype=np.int64)
        self._position_names = ['BTN', 'SB', 'BB', 'UTG', 'MP', 'CO']
        self._position_profit_sum = np.zeros(6, dtype=np.float64)
        self._position_hand_counts = np.zeros(6, dtype=np.int64)
        self._last_metric_position_profit_sum = np.zeros(6, dtype=np.float64)
        self._last_metric_position_hand_counts = np.zeros(6, dtype=np.int64)

        os.makedirs(checkpoint_dir, exist_ok=True)
        metrics_dir = os.path.dirname(self.metrics_output)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)

    def _write_metric(self, event: str, **payload) -> None:
        """Append one structured metric record for offline review agents."""
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            "total_hands": int(self.total_hands),
            **payload,
        }
        try:
            with open(self.metrics_output, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        except Exception as exc:
            logger.warning(f"写入 metrics 失败({self.metrics_output}): {exc}")

    def _action_distribution(self, counts: np.ndarray) -> Dict[str, float]:
        total = int(np.sum(counts))
        if total <= 0:
            return {name: 0.0 for name in self._action_names}
        return {
            name: float(counts[i] / total)
            for i, name in enumerate(self._action_names)
        }

    def _record_result_stats(self, res: Dict) -> None:
        """Aggregate per-hand diagnostics without retaining every hand result."""
        pos_idx = res.get('hero_position_index')
        try:
            pos_idx = int(pos_idx)
        except (TypeError, ValueError):
            pos_idx = None
        if pos_idx is None or not (0 <= pos_idx < len(self._position_names)):
            return

        profits = res.get('player_profits', {})
        profit = profits.get('Main_AI', profits.get('Player_0', 0.0))
        self._position_profit_sum[pos_idx] += float(profit)
        self._position_hand_counts[pos_idx] += 1

    def _position_profit_summary(self, profit_sum: np.ndarray, hand_counts: np.ndarray) -> Dict[str, Dict[str, float]]:
        summary = {}
        bb = max(float(self.big_blind), 1.0)
        for i, name in enumerate(self._position_names):
            hands = int(hand_counts[i])
            profit = float(profit_sum[i])
            summary[name] = {
                "hands": hands,
                "avg_profit": profit / max(hands, 1),
                "bb_per_100": (profit / bb) * 100.0 / max(hands, 1),
            }
        return summary

    def _append_completed_trajectory(
        self, res: Dict, traj: Dict, traj_buffer: Dict, results_buffer: List
    ) -> bool:
        """Append a whole completed hand without splitting it across PPO updates."""
        n_steps = len(traj.get('states', []))
        if n_steps <= 0:
            self._record_result_stats(res)
            return True

        cur_steps = len(traj_buffer.get('states', []))
        if cur_steps > 0 and (cur_steps + n_steps) > self.ppo_buffer_size:
            return False
        if cur_steps == 0 and n_steps > self.ppo_buffer_size:
            logger.warning(
                f"单手轨迹长度({n_steps})超过 ppo_buffer_size({self.ppo_buffer_size})，"
                "将允许本轮超限以避免切断 episode"
            )

        self._record_result_stats(res)
        for k in ('states', 'actions', 'old_probs'):
            traj_buffer[k].extend(traj.get(k, []))
        dones = [0.0] * n_steps
        rewards = [0.0] * n_steps
        dones[-1] = 1.0
        rewards[-1] = traj.get('final_reward', 0.0)
        traj_buffer['dones'].extend(dones)
        traj_buffer['rewards'].extend(rewards)

        for action_idx in traj.get('actions', []):
            ai = int(action_idx)
            if 0 <= ai < len(self._train_action_counts):
                self._train_action_counts[ai] += 1
        return True

    # ------------------------------------------------------------------
    #  Curriculum Learning
    # ------------------------------------------------------------------

    def _get_dynamic_opp_interval(self) -> int:
        """根据当前 Curriculum 阶段返回动态对手更新间隔。
        Phase 0: 不更新（全鱼）
        Phase 1: 100000（混合阶段，慢速适应）
        Phase 2: 500000（自对弈阶段，防止军备竞赛）
        """
        phase_intervals = [50000, 100000, 500000]  # Phase0/1/2 对应的间隔
        interval = phase_intervals[min(self._curriculum_phase, 2)]
        if self._opponent_scores and self.opponent_update_interval > 0:
            interval = min(interval, int(self.opponent_update_interval))
        return interval

    def _update_curriculum_phase(self):
        """根据训练进度自动切换对手难度阶段。

        Phase 0 (0-20%): 全鱼对手 → 学习基本策略
        Phase 1 (20-50%): 混合对手 → 过渡
        Phase 2 (50%+): 正常自对弈 → 充分博弈
        """
        if self.num_hands_target <= 0:
            return
        run_hands = max(0, self.total_hands - self._curriculum_base_hands)
        progress = run_hands / self.num_hands_target

        new_phase = self._curriculum_min_phase
        if progress >= self._curriculum_thresholds[1]:
            new_phase = max(new_phase, 2)
        elif progress >= self._curriculum_thresholds[0]:
            new_phase = max(new_phase, 1)
        new_phase = min(new_phase, self._curriculum_max_phase)

        if new_phase != self._curriculum_phase:
            old_phase = self._curriculum_phase
            self._curriculum_phase = new_phase
            self.opp_type_list = self._curriculum_opp_configs[new_phase]
            phase_names = ['Phase1(全鱼)', 'Phase2(混合)', 'Phase3(自对弈)']
            logger.info(
                f"[SEED RL] Curriculum 阶段切换: {phase_names[old_phase]} → {phase_names[new_phase]}, "
                f"hands={self.total_hands}, run_hands={run_hands}, progress={progress:.1%}, "
                f"opp_types={self.opp_type_list}, opp_interval={self._get_dynamic_opp_interval()}"
            )
            # 阶段切换时引入历史对手；没有历史池时才退回镜像同步。
            if new_phase >= 1:
                if self._opponent_pool:
                    self._update_opponent_from_pool()
                    logger.info("[SEED RL] 阶段切换时加载历史对手权重")
                else:
                    self.opp_model.set_weights(self.model.get_weights())
                    logger.info("[SEED RL] 阶段切换时同步对手权重（无历史池回退）")

    # ------------------------------------------------------------------
    #  推理调度
    # ------------------------------------------------------------------

    def _dispatch_inference(
        self, pipes: List, pending: Dict
    ) -> int:
        """
        超时积攒式批量推理调度。

        核心思路：循环收集 worker 请求，直到满足以下任一条件才触发 GPU 推理：
          1. batch 达到 inference_max_batch（立刻开火）
          2. batch >= inference_min_batch 且超时（开火）
          3. batch > 0 且等待超过 3× timeout（防死锁开火）
          4. 1 秒内无任何数据（退出）

        主 agent 用 self.model 推理，PPO 对手用独立的 self.opp_model 推理，
        彻底消除 set_weights 开销。
        """
        t0 = time.time()
        batch_items = []  # [(wid, player_name, state_vec), ...]

        while True:
            # 1. 扫描所有管道收集请求
            for wid, pipe in enumerate(pipes):
                try:
                    while pipe.poll(0):  # 非阻塞
                        msg = pipe.recv()
                        if msg[0] == 'infer':
                            _, w_id, player_name, state_vec = msg
                            pending[wid].append((player_name, state_vec))
                except (EOFError, BrokenPipeError):
                    continue

            # 2. 从 pending 组装 batch
            for wid in list(pending.keys()):
                while pending[wid] and len(batch_items) < self.inference_max_batch:
                    pname, sv = pending[wid].pop(0)
                    batch_items.append((wid, pname, sv))

            # 3. 判断 GPU 触发条件
            if len(batch_items) >= self.inference_max_batch:
                break  # 达到最大上限，立刻开火

            elapsed = time.time() - t0
            if len(batch_items) >= self.inference_min_batch and elapsed >= self.inference_timeout_s:
                break  # 满足最小批量且超时，开火

            if len(batch_items) > 0 and elapsed >= self.inference_timeout_s * 3:
                break  # 防死锁：虽然没达到最小批量，但等太久了

            if elapsed > 1.0:
                break  # 1 秒内无任何数据，退出

        if not batch_items:
            return 0

        # ---- GPU 批量推理 ----
        # 区分主 agent / PPO 对手 / Fish 对手（Curriculum 感知）
        main_items = []  # (original_index, wid, pname, sv)
        opp_ppo_items = defaultdict(list)  # model_idx -> PPO 对手 → GPU 推理
        opp_fish_items = []  # Fish 对手 → 随机动作

        for i, (wid, pname, sv) in enumerate(batch_items):
            if pname == "Main_AI":
                main_items.append((i, wid, pname, sv))
            else:
                # 根据 Curriculum 阶段判断该对手是 PPO 还是 Fish
                oi = int(pname.split('_')[-1]) - 1 if '_' in pname else 0
                oi = max(0, min(oi, len(self.opp_type_list) - 1))
                if self.opp_type_list[oi] == 'fish':
                    opp_fish_items.append((i, wid, pname, sv))
                else:
                    model_idx = oi % max(len(self._opponent_models), 1)
                    opp_ppo_items[model_idx].append((i, wid, pname, sv))

        # 预分配结果数组
        n = len(batch_items)
        action_probs = np.zeros((n, self.model.output_dim), dtype=np.float32)
        values = np.zeros(n, dtype=np.float32)

        # 主 agent 推理
        if main_items:
            main_states = np.array([it[3] for it in main_items], dtype=np.float32)
            m_probs, m_vals = self.model.predict_batch(main_states)
            for j, (oi, _, _, _) in enumerate(main_items):
                action_probs[oi] = m_probs[j]
                values[oi] = m_vals[j]

        # PPO 对手用独立模型推理 — 零权重切换开销
        for model_idx, items in opp_ppo_items.items():
            if not items:
                continue
            opp_states = np.array([it[3] for it in items], dtype=np.float32)
            model = self._opponent_models[model_idx]
            o_probs, o_vals = model.predict_batch(opp_states)
            for j, (oi, _, _, _) in enumerate(items):
                action_probs[oi] = o_probs[j]
                values[oi] = o_vals[j]

        # Fish 对手直接返回均匀随机概率（不走 GPU，零开销）
        if opp_fish_items:
            uniform_prob = 1.0 / self.model.output_dim
            for (oi, _, _, _) in opp_fish_items:
                action_probs[oi] = uniform_prob  # 均匀分布
                values[oi] = 0.0

        self._inference_batches += 1
        self._inference_items += len(batch_items)
        self._inference_main_items += len(main_items)
        self._inference_opp_items += sum(len(items) for items in opp_ppo_items.values())
        self._inference_fish_items += len(opp_fish_items)
        self._inference_seconds += time.time() - t0

        # 分发结果
        for i, (wid, pname, sv) in enumerate(batch_items):
            try:
                pipes[wid].send((action_probs[i], float(values[i])))
            except (BrokenPipeError, OSError):
                continue

        return len(batch_items)

    # ------------------------------------------------------------------
    #  轨迹收集
    # ------------------------------------------------------------------

    def _collect_trajectories(
        self, traj_queue: Queue, traj_buffer: Dict, results_buffer: List,
        alive_workers: set,
    ) -> int:
        """从 traj_queue 非阻塞读取完成的手牌。返回新收集的手牌数。
        
        关键：在吸取过程中实时检查 buffer 大小，一旦达到阈值立刻刹车，
        防止在高并发下一口气吸入几万条轨迹撑爆内存。
        """
        new_hands = 0

        while self._traj_overflow:
            if len(traj_buffer.get('states', [])) >= self.ppo_buffer_size:
                break
            res, traj = self._traj_overflow[0]
            if not self._append_completed_trajectory(res, traj, traj_buffer, results_buffer):
                break
            self._traj_overflow.pop(0)
            new_hands += 1

        while not traj_queue.empty():
            # 【背压拦截】buffer 已满，立刻停止吸取，留给训练消化
            if len(traj_buffer.get('states', [])) >= self.ppo_buffer_size:
                break

            try:
                msg = traj_queue.get_nowait()
            except Exception:
                break

            if msg[0] == 'traj':
                _, res, traj = msg
                if self._append_completed_trajectory(res, traj, traj_buffer, results_buffer):
                    new_hands += 1
                else:
                    self._traj_overflow.append((res, traj))
                    break

            elif msg[0] == 'done':
                wid = msg[1]
                alive_workers.discard(wid)

            elif msg[0] == 'error':
                _, wid, err_msg, tb = msg
                logger.warning(f"Worker {wid} 出错: {err_msg}\n{tb}")

        return new_hands

    # ------------------------------------------------------------------
    #  PPO 训练
    # ------------------------------------------------------------------

    def _do_ppo_train(self, traj_buffer: Dict, force: bool = False) -> Optional[float]:
        """如果 buffer 积累了足够数据，执行一次 PPO 训练并清空 buffer。"""
        n = len(traj_buffer['states'])
        if n < self.ppo_buffer_size and not force:
            return None
        if n <= 0:
            return None

        states = np.array(traj_buffer['states'], dtype=np.float32)
        actions = np.array(traj_buffer['actions'], dtype=np.int32)
        rewards = np.array(traj_buffer['rewards'], dtype=np.float32)
        dones = np.array(traj_buffer['dones'], dtype=np.float32)
        old_probs = np.array(traj_buffer['old_probs'], dtype=np.float32)

        # 清空 buffer
        for k in traj_buffer:
            traj_buffer[k] = []

        t0 = time.time()
        loss = self.model.train(states, actions, rewards, dones, old_probs)
        self._last_ppo_seconds = time.time() - t0
        self._last_ppo_steps = int(n)
        self.total_ppo_updates += 1
        return loss

    # ------------------------------------------------------------------
    #  评估
    # ------------------------------------------------------------------

    def _select_eval_action(
        self,
        probs: np.ndarray,
        legal_mask: np.ndarray,
        stochastic: bool,
        temperature: float = 0.5,
        epsilon: float = 0.02,
    ) -> int:
        masked = np.asarray(probs, dtype=np.float64) * np.asarray(legal_mask, dtype=np.float64)
        legal = np.asarray(legal_mask, dtype=np.float64) > 0
        if not np.any(legal):
            return 0
        total = float(masked.sum())
        if total <= 1e-8:
            masked = legal.astype(np.float64)
            total = float(masked.sum())
        dist = masked / max(total, 1e-8)
        if not stochastic:
            return int(np.argmax(dist))

        temp = max(float(temperature), 1e-3)
        logits = np.full_like(dist, -1e9, dtype=np.float64)
        logits[legal] = np.log(np.clip(dist[legal], 1e-8, 1.0)) / temp
        logits -= np.max(logits[legal])
        sampled = np.zeros_like(dist, dtype=np.float64)
        sampled[legal] = np.exp(logits[legal])
        sampled /= max(float(sampled.sum()), 1e-8)
        legal_uniform = legal.astype(np.float64) / max(float(np.sum(legal)), 1.0)
        sampled = (1.0 - epsilon) * sampled + epsilon * legal_uniform
        sampled /= max(float(sampled.sum()), 1e-8)
        return int(np.random.choice(ACTION_DIM, p=sampled))

    def _run_evaluation(
        self,
        num_games: int = 100,
        opponent_policy: str = "random",
        seed: Optional[int] = None,
    ) -> Dict:
        """简易评估：用当前模型 vs 随机对手打 num_games 手。

        修复说明：
        1. 移除 _preflop_filter：评估应反映模型真实输出，不叠加翻牌前过滤偏差
        2. 随机对手评估使用主模型 argmax；模型对手评估使用低温采样，
           避免双方确定性镜像导致 check/call 循环。
        3. 支持固定 seed 与 opponent_model 对手，降低随机鱼对手造成的乐观偏差
        4. 新增行动分布日志：打印 fold/call/raise 比例，便于诊断弃牌机器人问题
        注意：评估在主进程中串行执行（非多进程），保持简洁。
        """
        from ..model.state_representation import StateRepresentation
        from ..game.poker_game import PokerGame, Player
        from .env_worker import _legal_mask

        np_state = None
        if seed is not None:
            np_state = np.random.get_state()
            np.random.seed(int(seed))

        sr = StateRepresentation()
        wins, total_profit = 0, 0.0
        invalid_bets = 0
        invalid_actions = 0
        total_actions = 0
        truncated_hands = 0
        truncation_penalty_count = 0
        max_hand_actions_observed = 0
        max_actions_per_hand = 256
        # 行动分布统计（用于诊断弃牌机器人）
        action_names = ACTION_NAMES
        action_counts = [0] * ACTION_DIM
        invalid_action_counts = defaultdict(int)
        truncated_action_counts = []
        stochastic_model_eval = opponent_policy == "opponent_model"

        try:
            for _ in range(num_games):
                players_e = [Player("Main_AI", self.initial_stack, True)]
                for i in range(5):
                    players_e.append(Player(f"Opponent_{i + 1}", self.initial_stack, True))
                game = PokerGame(self.small_blind, self.big_blind, players_e)
                game.dealer_pos = np.random.randint(6)
                game.start_new_hand()

                hand_actions = 0
                while not game.is_hand_over():
                    hand_actions += 1
                    max_hand_actions_observed = max(max_hand_actions_observed, hand_actions)
                    if hand_actions > max_actions_per_hand:
                        truncated_hands += 1
                        invalid_bets += 1
                        truncation_penalty_count += 1
                        truncated_action_counts.append(hand_actions)
                        players_e[0].is_active = False
                        alive = [p for p in players_e if p.is_active]
                        if alive and game.pot > 0:
                            game._award_pot(alive[0])
                        if truncated_hands <= 5:
                            logger.warning(
                                f"[评估:{opponent_policy}] 单手超过 {max_actions_per_hand} 次动作，"
                                "已强制结束以避免评估卡死"
                            )
                        break

                    if game.is_round_over():
                        # proceed_to_next_round 返回 True = 手牌结束（摊牌或只剩1人）
                        # 返回 False = 仅进入下一轮（flop/turn/river），手牌继续
                        if game.proceed_to_next_round():
                            break
                    cp = game.next_player()
                    if not cp or not cp.is_active:
                        continue

                    if cp.name == "Main_AI":
                        state = sr.get_state(game, cp, getattr(self, 'opp_type_list', None))
                        ap, v = self.model.predict(state)
                        lm = _legal_mask(game, cp)
                        mp_ = ap * lm
                        s = float(np.sum(mp_))
                        # 归一化（不加 _preflop_filter，评估看真实策略）
                        mp_ = mp_ / s if s > 1e-8 else lm / (np.sum(lm) + 1e-8)
                        ai = self._select_eval_action(mp_, lm, stochastic=stochastic_model_eval)
                        action_counts[ai] += 1
                        av = np.zeros(ACTION_DIM, dtype=np.float32)
                        av[ai] = 1.0
                        hs = sr.calculate_hand_strength(cp, game)
                        act, amt = sr.decode_action(av, game, cp, hs)
                        total_actions += 1
                        if not game.handle_action(cp, act, amt):
                            invalid_actions += 1
                            invalid_bets += 1
                            invalid_action_counts[action_names[ai]] += 1
                            action_counts[0] += 1  # 无效动作降级为弃牌
                            action_counts[ai] -= 1
                            game.handle_action(cp, 'fold', None)
                    elif opponent_policy == "opponent_model":
                        state = sr.get_state(game, cp, getattr(self, 'opp_type_list', None))
                        ap, _ = self.opp_model.predict(state)
                        lm = _legal_mask(game, cp)
                        mp_ = ap * lm
                        s = float(np.sum(mp_))
                        mp_ = mp_ / s if s > 1e-8 else lm / (np.sum(lm) + 1e-8)
                        ai = self._select_eval_action(mp_, lm, stochastic=True)
                        av = np.zeros(ACTION_DIM, dtype=np.float32)
                        av[ai] = 1.0
                        hs = sr.calculate_hand_strength(cp, game)
                        act, amt = sr.decode_action(av, game, cp, hs)
                        if not game.handle_action(cp, act, amt):
                            game.handle_action(cp, 'fold', None)
                    else:
                        # 对手随机
                        lm = _legal_mask(game, cp)
                        p = lm / (np.sum(lm) + 1e-8)
                        ai = int(np.random.choice(ACTION_DIM, p=p))
                        av = np.zeros(ACTION_DIM, dtype=np.float32)
                        av[ai] = 1.0
                        hs = sr.calculate_hand_strength(cp, game)
                        act, amt = sr.decode_action(av, game, cp, hs)
                        if not game.handle_action(cp, act, amt):
                            game.handle_action(cp, 'fold', None)

                alive = [p for p in players_e if p.is_active]
                if len(alive) == 1 and game.pot > 0:
                    game._award_pot(alive[0])

                profit = players_e[0].stack - self.initial_stack
                total_profit += profit
                ws = [p.name for p in getattr(game, 'winners', [])]
                if not ws and hasattr(game, 'winner') and game.winner:
                    ws = [game.winner.name]
                if "Main_AI" in ws:
                    wins += 1
        finally:
            if np_state is not None:
                np.random.set_state(np_state)

        # 行动分布日志：帮助诊断弃牌机器人问题
        action_distribution = {}
        if total_actions > 0:
            action_distribution = {
                action_names[i]: action_counts[i] / total_actions
                for i in range(ACTION_DIM)
            }
            dist_str = ', '.join(
                f"{name}={value:.1%}"
                for name, value in action_distribution.items()
            )
            logger.info(f"[评估:{opponent_policy}] 行动分布 (总{total_actions}次): {dist_str}")
        else:
            logger.warning(f"[评估:{opponent_policy}] Main_AI 未执行任何行动，可能存在游戏逻辑问题")
        if truncated_hands > 0:
            logger.warning(
                f"[评估:{opponent_policy}] 共截断 {truncated_hands}/{num_games} 手牌 "
                f"({truncated_hands / max(num_games, 1):.1%})"
            )
        if invalid_actions > 0:
            invalid_dist = ', '.join(
                f"{name}={count}"
                for name, count in sorted(invalid_action_counts.items())
            )
            logger.warning(
                f"[评估:{opponent_policy}] Main_AI 实际非法动作 {invalid_actions} 次: {invalid_dist}"
            )

        avg_profit = total_profit / max(num_games, 1)
        bb_per_100 = (avg_profit / max(float(self.big_blind), 1.0)) * 100.0
        invalid_action_rate = invalid_actions / max(total_actions, 1)
        truncation_penalty_rate = truncation_penalty_count / max(total_actions, 1)
        invalid_or_truncated_rate = invalid_bets / max(total_actions, 1)
        invalid_action_distribution = {
            name: count / max(invalid_actions, 1)
            for name, count in sorted(invalid_action_counts.items())
        }
        truncated_action_avg = (
            sum(truncated_action_counts) / len(truncated_action_counts)
            if truncated_action_counts else 0.0
        )
        return {
            'win_rate': wins / max(num_games, 1),
            'avg_profit': avg_profit,
            'bb_per_100': bb_per_100,
            'invalid_bet_rate': invalid_action_rate,
            'invalid_action_count': invalid_actions,
            'invalid_action_rate': invalid_action_rate,
            'invalid_action_distribution': invalid_action_distribution,
            'invalid_or_truncated_rate': invalid_or_truncated_rate,
            'truncation_penalty_count': truncation_penalty_count,
            'truncation_penalty_rate': truncation_penalty_rate,
            'max_actions_per_hand': max_actions_per_hand,
            'max_hand_actions_observed': max_hand_actions_observed,
            'truncated_hand_action_avg': truncated_action_avg,
            'action_distribution': action_distribution,
            'total_profit': total_profit,
            'wins': wins,
            'total_games': num_games,
            'opponent_policy': opponent_policy,
            'selection_policy': "low_temp_sample" if stochastic_model_eval else "main_argmax",
            'seed': seed,
            'truncated_hands': truncated_hands,
            'truncated_hand_rate': truncated_hands / max(num_games, 1),
        }

    def _run_evaluation_suite(self, num_games: int = 100, seed: Optional[int] = None) -> Dict:
        """Evaluate against weak random opponents and current opponent-model opponents."""
        random_eval = self._run_evaluation(num_games, opponent_policy="random", seed=seed)
        opponent_seed = None if seed is None else int(seed) + 1
        opponent_eval = self._run_evaluation(
            num_games,
            opponent_policy="opponent_model",
            seed=opponent_seed,
        )
        result = dict(opponent_eval)
        result["primary_opponent_policy"] = "opponent_model"
        result["evaluations"] = {
            "random": random_eval,
            "opponent_model": opponent_eval,
        }
        return result

    # ------------------------------------------------------------------
    #  Checkpoint
    # ------------------------------------------------------------------

    def _add_checkpoints_to_opponent_pool(self, checkpoint_dir: str) -> int:
        """Add seed_rl_hands checkpoints from a directory to the opponent pool."""
        if not checkpoint_dir:
            return 0
        pattern = os.path.join(checkpoint_dir, "seed_rl_hands_*.actor.weights.h5")
        added = 0
        for idx_file in sorted(glob.glob(pattern))[-self._opponent_pool_max_size:]:
            base = idx_file.replace(".actor.weights.h5", "")
            if base not in self._opponent_pool:
                self._opponent_pool.append(base)
                added += 1
        if len(self._opponent_pool) > self._opponent_pool_max_size:
            self._opponent_pool = self._opponent_pool[-self._opponent_pool_max_size:]
        return added

    def _load_opponent_scores(self, path: Optional[str]) -> Dict[str, float]:
        """Load opponent bb/100 scores from JSONL matrix evaluation output."""
        if not path:
            return {}
        scores: Dict[str, float] = {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    opponent_path = row.get("opponent")
                    if opponent_path:
                        opponent_path = str(opponent_path)
                        if os.path.exists(opponent_path + ".actor.weights.h5"):
                            self._opponent_score_paths.append(opponent_path)
                    name = row.get("opponent_name")
                    if not name and opponent_path:
                        name = os.path.basename(opponent_path)
                    if name:
                        scores[str(name)] = float(row.get("bb_per_100", 0.0))
            logger.info(f"[SEED RL] 已加载 hard-opponent 评分 {len(scores)} 条: {path}")
        except Exception as exc:
            logger.warning(f"[SEED RL] 加载 hard-opponent 评分失败: {exc}")
        return scores

    def _load_opponent_roster(self, path: Optional[str]) -> List[Dict]:
        """Load fixed opponent buckets for ensemble slots.

        The roster intentionally matches checkpoints by basename so the same
        file works on local and server paths as long as the score file provides
        the checkpoint locations.
        """
        if not path:
            return []
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            buckets = data.get("buckets", data) if isinstance(data, dict) else data
            if not isinstance(buckets, list):
                raise ValueError("roster must be a list or {'buckets': [...]}")

            loaded = []
            for idx, bucket in enumerate(buckets):
                if not isinstance(bucket, dict):
                    raise ValueError(f"bucket #{idx} is not an object")
                names = bucket.get("opponents") or bucket.get("names") or []
                if isinstance(names, str):
                    names = [names]
                clean_names = [
                    os.path.basename(str(name).replace(".actor.weights.h5", ""))
                    for name in names
                    if str(name).strip()
                ]
                if not clean_names:
                    continue
                weight = 1.0
                if "weight" in bucket:
                    weight = max(0.0, float(bucket.get("weight", 1.0)))
                    self._opponent_roster_weighted = True
                loaded.append({
                    "name": str(bucket.get("name") or f"bucket_{idx}"),
                    "opponents": clean_names,
                    "weight": weight,
                })
            logger.info(
                "[SEED RL] 已加载固定对手 roster: "
                + ", ".join(
                    f"{bucket['name']}({len(bucket['opponents'])}, w={bucket['weight']:.2f})"
                    for bucket in loaded
                )
            )
            return loaded
        except Exception as exc:
            logger.warning(f"[SEED RL] 加载固定对手 roster 失败: {exc}")
            return []

    def _add_scored_opponents_to_pool(self) -> int:
        """Add explicit opponent checkpoint paths from the matrix score file."""
        added = 0
        for base in self._opponent_score_paths:
            if base not in self._opponent_pool:
                self._opponent_pool.append(base)
                added += 1
        if len(self._opponent_pool) > self._opponent_pool_max_size:
            self._opponent_pool = self._opponent_pool[-self._opponent_pool_max_size:]
        return added

    def _sample_opponent_checkpoint(self, candidates: List[str]) -> str:
        """Sample a checkpoint, preferring opponents with worse recent bb/100."""
        if not self._opponent_scores:
            pool_size = len(candidates)
            if pool_size >= 3 and random.random() < 0.7:
                return random.choice(candidates[-3:])
            return random.choice(candidates)

        scored_candidates = [
            path for path in candidates
            if os.path.basename(path) in self._opponent_scores
        ]
        if (
            scored_candidates
            and self.hard_opponent_scored_only_ratio > 0.0
            and random.random() < self.hard_opponent_scored_only_ratio
        ):
            candidates = scored_candidates

        negative_candidates = [
            path for path in candidates
            if self._opponent_scores.get(os.path.basename(path), 0.0) < 0.0
        ]
        positive_candidates = [
            path for path in candidates
            if self._opponent_scores.get(os.path.basename(path), 0.0) >= 0.0
        ]
        positive_ratio = self.hard_opponent_positive_ratio
        negative_ratio = min(self.hard_opponent_negative_ratio, max(0.0, 1.0 - positive_ratio))
        roll = random.random()
        if positive_candidates and positive_ratio > 0.0 and roll < positive_ratio:
            candidates = positive_candidates
        elif negative_candidates and negative_ratio > 0.0 and roll < positive_ratio + negative_ratio:
            candidates = negative_candidates

        weights = []
        for path in candidates:
            bb100 = self._opponent_scores.get(os.path.basename(path), 0.0)
            # Negative bb/100 means this opponent beats the current candidate.
            weights.append(math.exp(max(min(-bb100 / self.hard_opponent_temperature, 5.0), -5.0)))
        total = sum(weights)
        if total <= 0 or not math.isfinite(total):
            return random.choice(candidates)
        r = random.random() * total
        acc = 0.0
        for path, weight in zip(candidates, weights):
            acc += weight
            if acc >= r:
                return path
        return candidates[-1]

    def _bucket_candidates(self, candidates: List[str], bucket: Dict) -> List[str]:
        names = set(bucket.get("opponents", []))
        matched = [
            path for path in candidates
            if os.path.basename(path) in names
        ]
        return matched or candidates

    def _sample_roster_bucket(self, slot_idx: int) -> Dict:
        if not self._opponent_roster_weighted:
            return self._opponent_roster_buckets[
                slot_idx % len(self._opponent_roster_buckets)
            ]

        weights = [max(0.0, float(bucket.get("weight", 1.0))) for bucket in self._opponent_roster_buckets]
        total = sum(weights)
        if total <= 0 or not math.isfinite(total):
            return self._opponent_roster_buckets[
                slot_idx % len(self._opponent_roster_buckets)
            ]

        r = random.random() * total
        acc = 0.0
        for bucket, weight in zip(self._opponent_roster_buckets, weights):
            acc += weight
            if acc >= r:
                return bucket
        return self._opponent_roster_buckets[-1]

    def _sample_roster_paths(self, candidates: List[str]) -> List[str]:
        chosen_paths = []
        if not self._opponent_roster_buckets:
            available = list(candidates)
            for _ in range(len(self._opponent_models)):
                if not available:
                    available = list(candidates)
                chosen = self._sample_opponent_checkpoint(available)
                chosen_paths.append(chosen)
                if chosen in available and len(available) > 1:
                    available.remove(chosen)
            return chosen_paths

        for slot_idx in range(len(self._opponent_models)):
            bucket = self._sample_roster_bucket(slot_idx)
            bucket_candidates = self._bucket_candidates(candidates, bucket)
            chosen_paths.append(self._sample_opponent_checkpoint(bucket_candidates))
        return chosen_paths

    def _update_opponent_from_pool(self):
        """从历史对手池中抽取一个或多个版本加载到对手模型。

        安全阀机制：
        - 连续 3 次 win_rate < 0.2: 临时降级对手配置（增加鱼的比例）
        - 连续 2 次 win_rate < 0.25: 冻结对手池更新（不换更强的对手）
        抽取策略：无 score 时 70% 概率抽最近 3 个版本；有 score 时偏向亏损对手。
        如果对手池为空或加载失败，回退到同步当前模型权重。
        """
        wr = self._latest_win_rate
        bb100 = self._latest_eval_bb_per_100
        low_streak = self._low_win_eval_streak
        normal_config = self._curriculum_opp_configs[min(self._curriculum_phase, 2)]

        # 安全阀 1: 连续低胜率时才临时降级，避免单次评估噪声长期削弱对手池
        if low_streak >= 3 and wr < 0.2 and bb100 < 0.0 and self._curriculum_phase >= 2:
            degraded = ['ppo', 'fish', 'fish', 'fish', 'fish']
            if self.opp_type_list != degraded:
                self.opp_type_list = degraded
                logger.warning(
                    f"[安全阀] 连续低胜率({low_streak}次, 最新{wr:.3f}<0.2)，"
                    "Phase3临时降级为1PPO+4鱼"
                )
            return  # 降级后不更新对手权重

        # 安全阀 2: 连续偏低时冻结对手池更新，但不立刻削弱对手配置
        if low_streak >= 2 and wr < 0.25 and bb100 < 0.0:
            logger.info(
                f"[安全阀] 连续胜率偏低({low_streak}次, 最新{wr:.3f}<0.25)，冻结对手池更新"
            )
            if self.opp_type_list != normal_config:
                self.opp_type_list = normal_config
                logger.info(f"[安全阀] 保持正常对手配置，不因单次低分继续降级: {normal_config}")
            return  # 不更新对手权重

        # 胜率正常（>= 0.3），如果之前被降级过则恢复
        if self.opp_type_list != normal_config:
            self.opp_type_list = normal_config
            logger.info(f"[安全阀] 胜率正常({wr:.3f})，恢复正常对手配置: {normal_config}")

        if not self._opponent_pool:
            # 池为空，回退到同步当前权重
            for model in self._opponent_models:
                model.set_weights(self.model.get_weights())
            self._opponent_ensemble_paths = ["current"] * len(self._opponent_models)
            return

        # 加权抽取：最近的版本被抽中概率更高。恢复训练时优先排除
        # 当前 base checkpoint，避免把历史池训练退化为镜像自对弈。
        candidates = list(self._opponent_pool)
        if self._loaded_checkpoint_path and len(candidates) > 1:
            loaded_abs = os.path.abspath(self._loaded_checkpoint_path)
            non_current = [
                p for p in candidates
                if os.path.abspath(p) != loaded_abs
            ]
            if non_current:
                candidates = non_current

        pool_size = len(candidates)
        chosen_paths = self._sample_roster_paths(candidates)

        loaded_paths: List[str] = []
        for model_idx, (model, chosen) in enumerate(zip(self._opponent_models, chosen_paths)):
            try:
                model.load_weights(chosen)
                loaded_paths.append(chosen)
            except Exception as e:
                logger.warning(
                    f"[SEED RL] 加载历史对手失败(slot={model_idx}, path={chosen}): {e}，"
                    "该 slot 回退到当前权重"
                )
                model.set_weights(self.model.get_weights())
                loaded_paths.append("current")
        self._opponent_ensemble_paths = loaded_paths

        parts = []
        for idx, path in enumerate(loaded_paths):
            name = os.path.basename(path)
            score = self._opponent_scores.get(name)
            score_text = "" if score is None else f":{score:.1f}"
            parts.append(f"slot{idx}={name}{score_text}")
        logger.info(
            f"[SEED RL] 对手池抽取 ensemble_size={len(self._opponent_models)} "
            f"(pool_size={pool_size}, hard_neg_ratio={self.hard_opponent_negative_ratio:.2f}, "
            f"positive_ratio={self.hard_opponent_positive_ratio:.2f}, "
            f"scored_only_ratio={self.hard_opponent_scored_only_ratio:.2f}, "
            f"roster_buckets={len(self._opponent_roster_buckets)}): "
            + ", ".join(parts)
        )

    def save_checkpoint(self, is_best: bool = False, max_keep: int = 30) -> str:
        path = os.path.join(self.checkpoint_dir, f"seed_rl_hands_{self.total_hands}")
        self.model.save_weights(path)
        logger.info(f"Checkpoint 保存: {path}")
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "seed_rl_best")
            self.model.save_weights(best_path)
            logger.info(f"Best model 保存: {best_path}")

        # 添加到对手池
        self._opponent_pool.append(path)
        if len(self._opponent_pool) > self._opponent_pool_max_size:
            self._opponent_pool = self._opponent_pool[-self._opponent_pool_max_size:]

        # ---- 自动管理：按时间排序保留最新 max_keep 个，旧的归档到数据盘 ----
        try:
            pattern = os.path.join(self.checkpoint_dir, "seed_rl_hands_*")
            index_files = glob.glob(pattern + ".actor.weights.h5")
            # 按文件修改时间升序排列（最旧的在前）
            index_files.sort(key=lambda f: os.path.getmtime(f))
            if len(index_files) > max_keep:
                to_archive = index_files[:-max_keep]
                for idx_file in to_archive:
                    prefix = idx_file.replace(".actor.weights.h5", "")
                    base_name = os.path.basename(prefix)
                    for f in glob.glob(prefix + "*"):
                        dst = os.path.join(self._archive_dir, os.path.basename(f))
                        try:
                            shutil.move(f, dst)
                        except OSError:
                            try:
                                os.remove(f)
                            except OSError:
                                pass
                    logger.info(f"归档旧 checkpoint 到数据盘: {base_name}")
        except Exception as e:
            logger.warning(f"归档 checkpoint 时出错: {e}")
        return path

    def _run_checkpoint_validation(self, checkpoint_path: str) -> bool:
        if not self.checkpoint_validation_command:
            return True
        if self.total_hands < self.validation_start_hands:
            return True

        env = os.environ.copy()
        env.update({
            "SEED_RL_CHECKPOINT": checkpoint_path,
            "SEED_RL_CHECKPOINT_DIR": self.checkpoint_dir,
            "SEED_RL_TOTAL_HANDS": str(int(self.total_hands)),
            "SEED_RL_METRICS_OUTPUT": self.metrics_output,
        })
        start = time.time()
        logger.info(
            "[SEED RL] 运行 checkpoint validation @%s: %s",
            self.total_hands,
            self.checkpoint_validation_command,
        )
        result = subprocess.run(
            self.checkpoint_validation_command,
            shell=True,
            cwd=os.getcwd(),
            env=env,
        )
        elapsed = time.time() - start
        ok = result.returncode == 0
        self._write_metric(
            "checkpoint_validation",
            checkpoint=checkpoint_path,
            command=self.checkpoint_validation_command,
            return_code=int(result.returncode),
            ok=bool(ok),
            elapsed_seconds=float(elapsed),
        )
        if ok:
            logger.info("[SEED RL] checkpoint validation 通过: %s", checkpoint_path)
        else:
            logger.warning(
                "[SEED RL] checkpoint validation 失败(return_code=%s): %s",
                result.returncode,
                checkpoint_path,
            )
        return ok

    def _anchor_kl_guard_failed(self) -> bool:
        if self.max_anchor_kl is None:
            return False
        anchor_kl = float(getattr(self.model, "last_anchor_kl", 0.0))
        if anchor_kl > self.max_anchor_kl:
            self._anchor_kl_high_streak += 1
        else:
            self._anchor_kl_high_streak = 0
        if self._anchor_kl_high_streak <= 0:
            return False

        self._write_metric(
            "anchor_kl_guard",
            anchor_kl=anchor_kl,
            max_anchor_kl=float(self.max_anchor_kl),
            high_streak=int(self._anchor_kl_high_streak),
            patience=int(self.anchor_kl_patience),
        )
        if self._anchor_kl_high_streak >= self.anchor_kl_patience:
            logger.warning(
                "[SEED RL] anchor KL 连续过高: %.4f > %.4f, streak=%s/%s",
                anchor_kl,
                self.max_anchor_kl,
                self._anchor_kl_high_streak,
                self.anchor_kl_patience,
            )
            return True
        return False

    def _reference_kl_guard_failed(self) -> bool:
        if self.max_reference_kl is None:
            return False
        reference_kl = float(getattr(self.model, "last_reference_kl", 0.0))
        if reference_kl > self.max_reference_kl:
            self._reference_kl_high_streak += 1
        else:
            self._reference_kl_high_streak = 0
        if self._reference_kl_high_streak <= 0:
            return False

        self._write_metric(
            "reference_kl_guard",
            reference_kl=reference_kl,
            max_reference_kl=float(self.max_reference_kl),
            high_streak=int(self._reference_kl_high_streak),
            patience=int(self.reference_kl_patience),
        )
        if self._reference_kl_high_streak >= self.reference_kl_patience:
            logger.warning(
                "[SEED RL] reference KL 连续过高: %.4f > %.4f, streak=%s/%s",
                reference_kl,
                self.max_reference_kl,
                self._reference_kl_high_streak,
                self.reference_kl_patience,
            )
            return True
        return False

    def _retention_guard_failed(self) -> bool:
        return self._anchor_kl_guard_failed() or self._reference_kl_guard_failed()

    def _find_latest_checkpoint(self) -> Optional[str]:
        """查找最新的 SEED RL checkpoint（按文件名中的手数排序）。"""
        import re
        candidates = []
        for file in os.listdir(self.checkpoint_dir):
            if file.startswith('seed_rl_hands_') and file.endswith('.actor.weights.h5'):
                match = re.search(r'seed_rl_hands_(\d+)', file)
                if match:
                    hands = int(match.group(1))
                    base = os.path.join(self.checkpoint_dir, file.replace('.actor.weights.h5', ''))
                    candidates.append((hands, base))
        if not candidates:
            return None
        # 按手数降序，返回最新的
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def load_checkpoint(self, path: Optional[str] = None) -> bool:
        """加载 checkpoint 并恢复训练状态（total_hands、curriculum phase）。"""
        import re
        if path is None:
            # 优先加载最新的手数 checkpoint，而非 seed_rl_best
            path = self._find_latest_checkpoint()
            if path is None:
                # 回退到 seed_rl_best
                path = os.path.join(self.checkpoint_dir, "seed_rl_best")
        try:
            self.model.load_weights(path)
            self._loaded_checkpoint_path = path
            logger.info(f"Checkpoint 加载: {path}")

            # 从文件名解析 total_hands
            match = re.search(r'seed_rl_hands_(\d+)', path)
            if match:
                self.total_hands = int(match.group(1))
                logger.info(f"恢复训练进度: total_hands={self.total_hands}")
                # 恢复 Curriculum 阶段
                self._update_curriculum_phase()
                logger.info(f"恢复 Curriculum 阶段: Phase {self._curriculum_phase + 1}, "
                           f"对手配置={self.opp_type_list}")

            # 同步对手模型权重
            try:
                for model in self._opponent_models:
                    model.set_weights(self.model.get_weights())
                self._opponent_ensemble_paths = ["current"] * len(self._opponent_models)
                logger.info("对手模型权重已同步")
            except Exception:
                pass

            return True
        except Exception as e:
            logger.warning(f"加载 checkpoint 失败: {e}")
            return False

    # ------------------------------------------------------------------
    #  主训练循环
    # ------------------------------------------------------------------

    def train(self, num_hands: int):
        """SEED RL 主训练循环。

        1. 启动 N 个 env_worker 子进程
        2. 主线程交替：推理调度 + 轨迹收集 + PPO 训练
        3. 定期评估 + checkpoint
        """
        from .env_worker import env_worker_loop

        self.num_hands_target = num_hands
        self._curriculum_base_hands = int(self.total_hands)
        self._traj_overflow = []
        self._train_action_counts = np.zeros(ACTION_DIM, dtype=np.int64)
        self._last_metric_action_counts = np.zeros(ACTION_DIM, dtype=np.int64)
        self._position_profit_sum = np.zeros(6, dtype=np.float64)
        self._position_hand_counts = np.zeros(6, dtype=np.int64)
        self._last_metric_position_profit_sum = np.zeros(6, dtype=np.float64)
        self._last_metric_position_hand_counts = np.zeros(6, dtype=np.int64)

        run_target_hands = int(self.total_hands + num_hands)
        logger.info(
            f"[SEED RL] 训练开始: num_hands={num_hands}, "
            f"start_hands={self.total_hands}, target_total_hands={run_target_hands}, "
            f"workers={self.num_workers}, tables_per_worker={self.tables_per_worker}, "
            f"并发牌桌={self.num_workers * self.tables_per_worker}, "
            f"ppo_buffer={self.ppo_buffer_size}, "
            f"inference_batch=[{self.inference_min_batch}, {self.inference_max_batch}]"
        )
        _phase_intervals = [50000, 100000, 500000]  # 与 _get_dynamic_opp_interval 保持一致
        _intervals_str = "/".join(f"Phase{i+1}={v//1000}K" for i, v in enumerate(_phase_intervals))
        logger.info(
            f"[SEED RL] Curriculum Learning 已启用: "
            f"Phase1(0-20%全鱼) → Phase2(20-50%混合) → Phase3(50%+自对弈), "
            f"dynamic_opp_interval=[{_intervals_str}], "
            f"min_phase=Phase{self._curriculum_min_phase + 1}, "
            f"max_phase=Phase{self._curriculum_max_phase + 1}, "
            f"历史对手池=开启(max_size={self._opponent_pool_max_size}), "
            f"opponent_ensemble_size={len(self._opponent_models)}, "
            f"hard_neg_ratio={self.hard_opponent_negative_ratio:.2f}, "
            f"positive_ratio={self.hard_opponent_positive_ratio:.2f}, "
            f"scored_only_ratio={self.hard_opponent_scored_only_ratio:.2f}, "
            f"roster_buckets={len(self._opponent_roster_buckets)}, "
            f"hard_temperature={self.hard_opponent_temperature:.1f}, "
            f"archive_dir={self._archive_dir}"
        )

        # 初始化对手模型权重并预热
        for model in self._opponent_models:
            model.set_weights(self.model.get_weights())
        self._opponent_ensemble_paths = ["current"] * len(self._opponent_models)

        # 初始化对手池：扫描当前 run 与外部加载 checkpoint 的目录。
        added_current = self._add_checkpoints_to_opponent_pool(self.checkpoint_dir)
        added_loaded = 0
        if self._loaded_checkpoint_path:
            loaded_dir = os.path.dirname(self._loaded_checkpoint_path)
            if os.path.abspath(loaded_dir) != os.path.abspath(self.checkpoint_dir):
                added_loaded = self._add_checkpoints_to_opponent_pool(loaded_dir)
        added_scored = self._add_scored_opponents_to_pool()
        if self._opponent_pool:
            logger.info(
                f"[SEED RL] 对手池已加载 {len(self._opponent_pool)} 个历史版本 "
                f"(current_dir+{added_current}, loaded_dir+{added_loaded}, "
                f"score_file+{added_scored})"
            )
            if self._curriculum_phase >= 1:
                self._update_opponent_from_pool()
                logger.info("[SEED RL] 启动阶段加载历史对手权重")

        # 模型预热（主模型 + 对手模型）
        self.model.warmup()
        for model in self._opponent_models:
            model.warmup()

        # 计算每个 worker 需要跑的手牌数
        hands_per_worker = max(1, num_hands // self.num_workers)
        extra = num_hands - hands_per_worker * self.num_workers

        game_cfg = {
            'small_blind': self.small_blind,
            'big_blind': self.big_blind,
            'initial_stack': self.initial_stack,
            'allin_eq_threshold': self.allin_eq_threshold,
            'allin_bonus_cap': self.allin_bonus_cap,
            'tables_per_worker': self.tables_per_worker,
        }

        # 创建通信管道
        main_pipes = []   # 主进程端
        worker_pipes = []  # Worker 端
        for _ in range(self.num_workers):
            parent_conn, child_conn = Pipe()
            main_pipes.append(parent_conn)
            worker_pipes.append(child_conn)

        traj_queue = Queue()

        # 启动 worker 进程
        workers = []
        alive_workers = set()
        for wid in range(self.num_workers):
            h = hands_per_worker + (1 if wid < extra else 0)
            p = Process(
                target=env_worker_loop,
                args=(wid, worker_pipes[wid], traj_queue, game_cfg,
                      self._worker_opp_type_list, h),
                daemon=True,
            )
            p.start()
            workers.append(p)
            alive_workers.add(wid)

        logger.info(f"[SEED RL] {len(workers)} 个 worker 进程已启动")

        # 训练状态
        traj_buffer = defaultdict(list)
        results_buffer = []
        pending = defaultdict(list)  # wid -> [(pname, state_vec), ...]

        training_start = time.time()
        training_start_hands = self.total_hands  # 记录 resume 后的起始手数
        last_eval_hand = training_start_hands
        last_ckpt_hand = training_start_hands
        last_opp_update_hand = training_start_hands
        last_metric_time = training_start
        last_metric_hands = training_start_hands
        last_saved_checkpoint_path: Optional[str] = None
        last_validated_checkpoint_path: Optional[str] = None
        stopped_by_validation = False
        stopped_by_anchor_guard = False

        self._write_metric(
            "train_start",
            target_hands=int(num_hands),
            start_hands=int(training_start_hands),
            target_total_hands=int(training_start_hands + num_hands),
            workers=int(self.num_workers),
            tables_per_worker=int(self.tables_per_worker),
            concurrent_tables=int(self.num_workers * self.tables_per_worker),
            ppo_buffer_size=int(self.ppo_buffer_size),
            skip_final_buffer_train=bool(self.skip_final_buffer_train),
            inference_min_batch=int(self.inference_min_batch),
            inference_max_batch=int(self.inference_max_batch),
            small_blind=int(self.small_blind),
            big_blind=int(self.big_blind),
            initial_stack=int(self.initial_stack),
            anchor_kl_coef=float(getattr(self.model, "anchor_kl_coef", 0.0)),
            max_anchor_kl=float(self.max_anchor_kl) if self.max_anchor_kl is not None else None,
            anchor_kl_patience=int(self.anchor_kl_patience),
            reference_distill_file=getattr(self.model, "reference_distill_file", None),
            reference_distill_rows=(
                int(len(getattr(self.model, "reference_states", [])))
                if getattr(self.model, "reference_states", None) is not None else 0
            ),
            reference_distill_coef=float(getattr(self.model, "reference_distill_coef", 0.0)),
            reference_distill_batch_size=int(getattr(self.model, "reference_distill_batch_size", 0)),
            max_reference_kl=float(self.max_reference_kl) if self.max_reference_kl is not None else None,
            reference_kl_patience=int(self.reference_kl_patience),
            opponent_ensemble_size=int(len(self._opponent_models)),
            opponent_ensemble=list(self._opponent_ensemble_paths),
            opponent_roster_buckets=[
                bucket.get("name") for bucket in self._opponent_roster_buckets
            ],
        )

        def log_ppo_update(loss: float, reason: str) -> None:
            nonlocal last_metric_time, last_metric_hands
            now = time.time()
            elapsed = now - training_start
            hps = (self.total_hands - training_start_hands) / max(elapsed, 1e-6)
            interval_hps = (self.total_hands - last_metric_hands) / max(now - last_metric_time, 1e-6)
            avg_batch = self._inference_items / max(self._inference_batches, 1)
            cumulative_action_dist = self._action_distribution(self._train_action_counts)
            interval_action_counts = self._train_action_counts - self._last_metric_action_counts
            interval_action_dist = self._action_distribution(interval_action_counts)
            cumulative_position_profit = self._position_profit_summary(
                self._position_profit_sum, self._position_hand_counts
            )
            interval_position_profit = self._position_profit_summary(
                self._position_profit_sum - self._last_metric_position_profit_sum,
                self._position_hand_counts - self._last_metric_position_hand_counts,
            )
            interval_action_summary = ", ".join(
                f"{name}={interval_action_dist[name]:.1%}"
                for name in self._action_names
            )
            interval_position_summary = ", ".join(
                f"{name}={interval_position_profit[name]['bb_per_100']:.1f}bb/100"
                for name in self._position_names
            )
            logger.info(
                f"[SEED RL] hands={self.total_hands}/{run_target_hands}, "
                f"PPO更新#{self.total_ppo_updates}, loss={loss:.6f}, "
                f"速度={hps:.1f} hands/s, 区间速度={interval_hps:.1f} hands/s, "
                f"推理batch均值={avg_batch:.1f}, PPO耗时={self._last_ppo_seconds:.3f}s, "
                f"KL={getattr(self.model, 'last_kl', 0.0):.4f}, "
                f"anchor_KL={getattr(self.model, 'last_anchor_kl', 0.0):.4f}, "
                f"reference_KL={getattr(self.model, 'last_reference_kl', 0.0):.4f}, "
                f"区间动作=[{interval_action_summary}], "
                f"位置收益=[{interval_position_summary}]"
            )
            self._write_metric(
                "ppo_update",
                reason=reason,
                target_hands=int(num_hands),
                start_hands=int(training_start_hands),
                target_total_hands=int(run_target_hands),
                ppo_updates=int(self.total_ppo_updates),
                loss=float(loss),
                hps=float(hps),
                interval_hps=float(interval_hps),
                ppo_seconds=float(self._last_ppo_seconds),
                ppo_steps=int(self._last_ppo_steps),
                last_kl=float(getattr(self.model, "last_kl", 0.0)),
                anchor_kl=float(getattr(self.model, "last_anchor_kl", 0.0)),
                anchor_kl_coef=float(getattr(self.model, "anchor_kl_coef", 0.0)),
                reference_kl=float(getattr(self.model, "last_reference_kl", 0.0)),
                reference_distill_coef=float(getattr(self.model, "reference_distill_coef", 0.0)),
                reference_distill_batch_size=int(getattr(self.model, "reference_distill_batch_size", 0)),
                reference_distill_rows=(
                    int(len(getattr(self.model, "reference_states", [])))
                    if getattr(self.model, "reference_states", None) is not None else 0
                ),
                inference_batches=int(self._inference_batches),
                inference_items=int(self._inference_items),
                inference_main_items=int(self._inference_main_items),
                inference_opp_items=int(self._inference_opp_items),
                inference_fish_items=int(self._inference_fish_items),
                inference_seconds=float(self._inference_seconds),
                avg_inference_batch=float(avg_batch),
                curriculum_phase=int(self._curriculum_phase + 1),
                opponent_types=list(self.opp_type_list),
                opponent_ensemble_size=int(len(self._opponent_models)),
                opponent_ensemble=list(self._opponent_ensemble_paths),
                opponent_roster_buckets=[
                    bucket.get("name") for bucket in self._opponent_roster_buckets
                ],
                train_action_distribution=cumulative_action_dist,
                train_action_distribution_interval=interval_action_dist,
                position_profit=cumulative_position_profit,
                position_profit_interval=interval_position_profit,
            )
            self._last_metric_action_counts = self._train_action_counts.copy()
            self._last_metric_position_profit_sum = self._position_profit_sum.copy()
            self._last_metric_position_hand_counts = self._position_hand_counts.copy()
            last_metric_time = now
            last_metric_hands = self.total_hands

        try:
            while alive_workers or not traj_queue.empty() or self._traj_overflow:
                # === 背压机制：buffer 超过阈值时优先训练，暂停推理 ===
                buf_size = len(traj_buffer.get('states', []))
                if buf_size >= self.ppo_buffer_size:
                    # 先把队列里剩余的轨迹全部收完
                    self._collect_trajectories(
                        traj_queue, traj_buffer, results_buffer, alive_workers
                    )
                    # 循环训练直到 buffer 消化完毕
                    while len(traj_buffer.get('states', [])) >= self.ppo_buffer_size:
                        loss = self._do_ppo_train(traj_buffer)
                        if loss is not None:
                            log_ppo_update(loss, reason="backpressure")
                            if self._retention_guard_failed():
                                stopped_by_anchor_guard = True
                                break
                    if stopped_by_anchor_guard:
                        break
                    # 训练完毕，继续正常循环

                # 1. 推理调度
                inferred = self._dispatch_inference(main_pipes, pending)

                # 2. 收集轨迹
                new_hands = self._collect_trajectories(
                    traj_queue, traj_buffer, results_buffer, alive_workers
                )
                self.total_hands += new_hands

                # 3. PPO 训练（常规路径）
                buf_size = len(traj_buffer.get('states', []))
                overflow_boundary = bool(self._traj_overflow) and buf_size > 0
                if buf_size >= self.ppo_buffer_size or overflow_boundary:
                    loss = self._do_ppo_train(traj_buffer, force=overflow_boundary)
                    if loss is not None:
                        log_ppo_update(loss, reason="overflow_boundary" if overflow_boundary else "regular")
                        if self._retention_guard_failed():
                            stopped_by_anchor_guard = True
                            logger.warning("[SEED RL] retention KL guard 触发，停止训练")
                            break

                # 4. 定期评估（与 checkpoint 间隔对齐，避免阻塞训练流水线）
                # 评估在主线程串行执行，会阻塞推理调度导致所有 worker 停滞
                eval_interval_actual = max(self.eval_interval, self.checkpoint_interval)
                if (self.total_hands - last_eval_hand) >= eval_interval_actual and self.total_hands > 0:
                    last_eval_hand = self.total_hands
                    eval_seed = None
                    if self.eval_seed is not None:
                        eval_seed = int(self.eval_seed) + int(self.total_hands)
                    eval_res = self._run_evaluation_suite(self.num_eval_games, seed=eval_seed)
                    score = eval_res['avg_profit']
                    logger.info(
                        f"[SEED RL] 评估@{self.total_hands}({eval_res.get('primary_opponent_policy')}): "
                        f"胜率={eval_res['win_rate']:.3f}, "
                        f"平均收益={score:.2f}, "
                        f"bb/100={eval_res.get('bb_per_100', 0.0):.1f}, "
                        f"无效下注率={eval_res['invalid_bet_rate']:.3f}, "
                        f"截断率={eval_res.get('truncated_hand_rate', 0.0):.3f}"
                    )
                    self._write_metric(
                        "evaluation",
                        eval_games=int(self.num_eval_games),
                        win_rate=float(eval_res["win_rate"]),
                        avg_profit=float(eval_res["avg_profit"]),
                        bb_per_100=float(eval_res.get("bb_per_100", 0.0)),
                        invalid_bet_rate=float(eval_res["invalid_bet_rate"]),
                        truncated_hand_rate=float(eval_res.get("truncated_hand_rate", 0.0)),
                        truncated_hands=int(eval_res.get("truncated_hands", 0)),
                        action_distribution=eval_res.get("action_distribution", {}),
                        evaluations=eval_res.get("evaluations", {}),
                        primary_opponent_policy=eval_res.get("primary_opponent_policy"),
                        selection_policy=eval_res.get("selection_policy"),
                        eval_seed=eval_seed,
                        best_score=float(self.best_score) if math.isfinite(self.best_score) else None,
                    )
                    self._latest_win_rate = eval_res['win_rate']
                    self._latest_eval_bb_per_100 = float(eval_res.get("bb_per_100", 0.0))
                    if self._latest_win_rate < 0.25 and self._latest_eval_bb_per_100 < 0.0:
                        self._low_win_eval_streak += 1
                    elif self._latest_win_rate >= 0.30 or self._latest_eval_bb_per_100 >= 0.0:
                        self._low_win_eval_streak = 0
                    logger.info(
                        f"[安全阀] 低胜率评估连续次数={self._low_win_eval_streak}, "
                        f"latest_win_rate={self._latest_win_rate:.3f}, "
                        f"latest_bb_per_100={self._latest_eval_bb_per_100:.1f}"
                    )
                    if score > self.best_score and eval_res['invalid_bet_rate'] < 0.5:
                        self.best_score = score
                        self.save_checkpoint(is_best=True)

                # 5. 定期保存 checkpoint
                if (self.total_hands - last_ckpt_hand) >= self.checkpoint_interval and self.total_hands > 0:
                    last_ckpt_hand = self.total_hands
                    checkpoint_path = self.save_checkpoint(is_best=False)
                    last_saved_checkpoint_path = checkpoint_path
                    validation_ok = self._run_checkpoint_validation(checkpoint_path)
                    last_validated_checkpoint_path = checkpoint_path
                    if not validation_ok and self.stop_on_validation_failure:
                        stopped_by_validation = True
                        logger.warning("[SEED RL] validation 失败，按配置停止训练")
                        break

                # 6. Curriculum Learning 阶段检查 + 对手权重更新
                self._update_curriculum_phase()

                # 动态对手更新间隔：Phase越高，间隔越长（防止军备竞赛）
                dynamic_interval = self._get_dynamic_opp_interval()
                if (self.total_hands - last_opp_update_hand) >= dynamic_interval and self.total_hands > 0:
                    last_opp_update_hand = self.total_hands
                    # Phase 0 全鱼模式不需要同步权重（鱼用随机策略）
                    if self._curriculum_phase >= 1:
                        self._update_opponent_from_pool()
                        logger.info(f"[SEED RL] 对手权重已更新 @{self.total_hands} (interval={dynamic_interval})")

                # 短暂 sleep 避免 CPU 空转
                if inferred == 0 and new_hands == 0:
                    time.sleep(0.0001)

                if stopped_by_anchor_guard:
                    break

        except KeyboardInterrupt:
            logger.info("[SEED RL] 收到中断信号，正在清理...")
        finally:
            # 清理 worker
            for p in workers:
                if p.is_alive():
                    p.terminate()
            for p in workers:
                p.join(timeout=5)
            for pipe in main_pipes:
                pipe.close()
            for pipe in worker_pipes:
                pipe.close()

        # 训练结束后处理剩余 buffer（仅当数据足够多时才训练，避免小样本灾难性过拟合）
        remaining = len(traj_buffer.get('states', []))
        min_remaining = max(8192, self.model.batch_size)  # 小尾批高方差，宁可丢弃
        if stopped_by_validation or stopped_by_anchor_guard:
            if remaining > 0:
                logger.info(
                    f"[SEED RL] 跳过剩余 buffer: {remaining} steps，"
                    "因为训练已被 validation/retention guard 停止"
                )
                for k in traj_buffer:
                    traj_buffer[k] = []
        elif self.skip_final_buffer_train:
            if remaining > 0:
                logger.info(
                    f"[SEED RL] 按配置跳过剩余 buffer: {remaining} steps "
                    "(skip_final_buffer_train=True)"
                )
                for k in traj_buffer:
                    traj_buffer[k] = []
        elif remaining >= min_remaining:
            logger.info(f"[SEED RL] 处理剩余 buffer: {remaining} steps")
            old_buf = self.ppo_buffer_size
            self.ppo_buffer_size = 1
            loss = self._do_ppo_train(traj_buffer, force=True)
            if loss is not None:
                self._write_metric(
                    "ppo_update",
                    reason="final_buffer",
                    target_hands=int(num_hands),
                    ppo_updates=int(self.total_ppo_updates),
                    loss=float(loss),
                    ppo_seconds=float(self._last_ppo_seconds),
                    ppo_steps=int(self._last_ppo_steps),
                    last_kl=float(getattr(self.model, "last_kl", 0.0)),
                    anchor_kl=float(getattr(self.model, "last_anchor_kl", 0.0)),
                    anchor_kl_coef=float(getattr(self.model, "anchor_kl_coef", 0.0)),
                    reference_kl=float(getattr(self.model, "last_reference_kl", 0.0)),
                    reference_distill_coef=float(getattr(self.model, "reference_distill_coef", 0.0)),
                )
            self.ppo_buffer_size = old_buf
        elif remaining > 0:
            logger.info(f"[SEED RL] 丢弃剩余 buffer: {remaining} steps < {min_remaining}（太少会导致过拟合）")

        elapsed = time.time() - training_start
        hps = (self.total_hands - training_start_hands) / max(elapsed, 1e-6)
        logger.info(
            f"[SEED RL] 训练完成: "
            f"总手牌={self.total_hands}, 总PPO更新={self.total_ppo_updates}, "
            f"耗时={elapsed:.1f}s, 速度={hps:.1f} hands/s"
        )
        self._write_metric(
            "train_complete",
            start_hands=int(training_start_hands),
            target_hands=int(num_hands),
            target_total_hands=int(run_target_hands),
            ppo_updates=int(self.total_ppo_updates),
            elapsed_seconds=float(elapsed),
            hps=float(hps),
            inference_batches=int(self._inference_batches),
            inference_items=int(self._inference_items),
            avg_inference_batch=float(self._inference_items / max(self._inference_batches, 1)),
            best_score=float(self.best_score) if math.isfinite(self.best_score) else None,
        )

        # 最终 checkpoint
        if last_saved_checkpoint_path and self.total_hands == last_ckpt_hand:
            final_checkpoint_path = last_saved_checkpoint_path
        else:
            final_checkpoint_path = self.save_checkpoint(is_best=False)
        if (
            not stopped_by_validation
            and final_checkpoint_path != last_validated_checkpoint_path
        ):
            self._run_checkpoint_validation(final_checkpoint_path)

        # 最终评估
        if not self.skip_final_eval:
            logger.info("[SEED RL] 最终评估...")
            final_seed = None
            if self.eval_seed is not None:
                final_seed = int(self.eval_seed) + int(self.total_hands) + 999
            final_eval = self._run_evaluation_suite(self.num_eval_games, seed=final_seed)
            logger.info(
                f"[SEED RL] 最终评估结果: "
                f"primary={final_eval.get('primary_opponent_policy')}, "
                f"胜率={final_eval['win_rate']:.3f}, "
                f"平均收益={final_eval['avg_profit']:.2f}, "
                f"bb/100={final_eval.get('bb_per_100', 0.0):.1f}, "
                f"无效下注率={final_eval['invalid_bet_rate']:.3f}"
            )
            self._write_metric(
                "final_evaluation",
                eval_games=int(self.num_eval_games),
                win_rate=float(final_eval["win_rate"]),
                avg_profit=float(final_eval["avg_profit"]),
                bb_per_100=float(final_eval.get("bb_per_100", 0.0)),
                invalid_bet_rate=float(final_eval["invalid_bet_rate"]),
                truncated_hand_rate=float(final_eval.get("truncated_hand_rate", 0.0)),
                truncated_hands=int(final_eval.get("truncated_hands", 0)),
                action_distribution=final_eval.get("action_distribution", {}),
                evaluations=final_eval.get("evaluations", {}),
                primary_opponent_policy=final_eval.get("primary_opponent_policy"),
                selection_policy=final_eval.get("selection_policy"),
                eval_seed=final_seed,
            )
