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
import threading
import numpy as np
import multiprocessing as mp
from multiprocessing import Queue, Process, Pipe
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

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
        inference_timeout_s: float = 0.001,
        ppo_buffer_size: int = 4096,
        skip_final_eval: bool = False,
        allin_eq_threshold: float = 0.65,
        allin_bonus_cap: float = 0.80,
        tables_per_worker: int = 8,
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
        self.allin_eq_threshold = allin_eq_threshold
        self.allin_bonus_cap = allin_bonus_cap
        self.tables_per_worker = tables_per_worker

        # 训练统计
        self.total_hands = 0
        self.total_ppo_updates = 0
        self.best_score = -float('inf')

        # 对手类型：前 4 个是 PPO，第 5 个是 Fish
        self.opp_type_list = ['ppo', 'ppo', 'ppo', 'ppo', 'fish']

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
        self._opp_weights = None  # 废弃旧的权重快照变量

        os.makedirs(checkpoint_dir, exist_ok=True)

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
        # 区分主 agent 和 PPO 对手
        main_items = []  # (original_index, wid, pname, sv)
        opp_items = []
        for i, (wid, pname, sv) in enumerate(batch_items):
            if pname == "Main_AI":
                main_items.append((i, wid, pname, sv))
            else:
                opp_items.append((i, wid, pname, sv))

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
        if opp_items:
            opp_states = np.array([it[3] for it in opp_items], dtype=np.float32)
            o_probs, o_vals = self.opp_model.predict_batch(opp_states)
            for j, (oi, _, _, _) in enumerate(opp_items):
                action_probs[oi] = o_probs[j]
                values[oi] = o_vals[j]

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
        """从 traj_queue 非阻塞读取完成的手牌。返回新收集的手牌数。"""
        new_hands = 0
        while not traj_queue.empty():
            try:
                msg = traj_queue.get_nowait()
            except Exception:
                break

            if msg[0] == 'traj':
                _, res, traj = msg
                results_buffer.append(res)
                # 轨迹写入 PPO buffer
                n_steps = len(traj.get('states', []))
                if n_steps > 0:
                    for k in ('states', 'actions', 'old_probs'):
                        traj_buffer[k].extend(traj.get(k, []))
                    # 构建 dones 和 rewards
                    dones = [0.0] * n_steps
                    rewards = [0.0] * n_steps
                    dones[-1] = 1.0
                    rewards[-1] = traj.get('final_reward', 0.0)
                    traj_buffer['dones'].extend(dones)
                    traj_buffer['rewards'].extend(rewards)
                new_hands += 1

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

    def _do_ppo_train(self, traj_buffer: Dict) -> Optional[float]:
        """如果 buffer 积累了足够数据，执行一次 PPO 训练并清空 buffer。"""
        n = len(traj_buffer['states'])
        if n < self.ppo_buffer_size:
            return None

        states = np.array(traj_buffer['states'], dtype=np.float32)
        actions = np.array(traj_buffer['actions'], dtype=np.int32)
        rewards = np.array(traj_buffer['rewards'], dtype=np.float32)
        dones = np.array(traj_buffer['dones'], dtype=np.float32)
        old_probs = np.array(traj_buffer['old_probs'], dtype=np.float32)

        # 清空 buffer
        for k in traj_buffer:
            traj_buffer[k] = []

        loss = self.model.train(states, actions, rewards, dones, old_probs)
        self.total_ppo_updates += 1
        return loss

    # ------------------------------------------------------------------
    #  评估
    # ------------------------------------------------------------------

    def _run_evaluation(self, num_games: int = 100) -> Dict:
        """简易评估：用当前模型 vs Fish 对手打 num_games 手。
        
        注意：评估在主进程中串行执行（非多进程），保持简洁。
        """
        from ..model.state_representation import StateRepresentation
        from ..game.poker_game import PokerGame, Player

        sr = StateRepresentation()
        wins, total_profit = 0, 0.0
        invalid_bets = 0
        total_actions = 0

        for _ in range(num_games):
            players_e = [Player("Main_AI", self.initial_stack, True)]
            for i in range(5):
                players_e.append(Player(f"Opponent_{i + 1}", self.initial_stack, True))
            game = PokerGame(self.small_blind, self.big_blind, players_e)
            game.dealer_pos = np.random.randint(6)
            game.start_new_hand()

            while not game.is_hand_over():
                if game.is_round_over():
                    if not game.proceed_to_next_round():
                        break
                cp = game.next_player()
                if not cp or not cp.is_active:
                    continue

                if cp.name == "Main_AI":
                    state = sr.get_state(game, cp)
                    ap, v = self.model.predict(state)
                    from .env_worker import _legal_mask, _preflop_filter
                    lm = _legal_mask(game, cp)
                    mp_ = ap * lm
                    s = float(np.sum(mp_))
                    mp_ = mp_ / s if s > 1e-8 else lm / (np.sum(lm) + 1e-8)
                    hs = sr.calculate_hand_strength(cp, game)
                    mp_ = _preflop_filter(mp_, game, cp, hs)
                    ai = int(np.random.choice(5, p=mp_))
                    av = np.zeros(5, dtype=np.float32)
                    av[ai] = 1.0
                    act, amt = sr.decode_action(av, game, cp, hs)
                    total_actions += 1
                    if not game.handle_action(cp, act, amt):
                        invalid_bets += 1
                        game.handle_action(cp, 'fold', None)
                else:
                    # 对手随机
                    from .env_worker import _legal_mask as _lm
                    lm = _lm(game, cp)
                    p = lm / (np.sum(lm) + 1e-8)
                    ai = int(np.random.choice(5, p=p))
                    av = np.zeros(5, dtype=np.float32)
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

        return {
            'win_rate': wins / max(num_games, 1),
            'avg_profit': total_profit / max(num_games, 1),
            'invalid_bet_rate': invalid_bets / max(total_actions, 1),
            'total_games': num_games,
        }

    # ------------------------------------------------------------------
    #  Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, is_best: bool = False):
        path = os.path.join(self.checkpoint_dir, f"seed_rl_hands_{self.total_hands}")
        self.model.save_weights(path)
        logger.info(f"Checkpoint 保存: {path}")
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "seed_rl_best")
            self.model.save_weights(best_path)
            logger.info(f"Best model 保存: {best_path}")

    def load_checkpoint(self, path: Optional[str] = None) -> bool:
        if path is None:
            path = os.path.join(self.checkpoint_dir, "seed_rl_best")
        try:
            self.model.load_weights(path)
            logger.info(f"Checkpoint 加载: {path}")
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

        logger.info(
            f"[SEED RL] 训练开始: num_hands={num_hands}, "
            f"workers={self.num_workers}, tables_per_worker={self.tables_per_worker}, "
            f"并发牌桌={self.num_workers * self.tables_per_worker}, "
            f"ppo_buffer={self.ppo_buffer_size}, "
            f"inference_batch=[{self.inference_min_batch}, {self.inference_max_batch}]"
        )

        # 初始化对手模型权重并预热
        self.opp_model.set_weights(self.model.get_weights())

        # 模型预热（主模型 + 对手模型）
        self.model.warmup()
        self.opp_model.warmup()

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
                      self.opp_type_list, h),
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

        last_eval_hand = 0
        last_ckpt_hand = 0
        last_opp_update_hand = 0
        training_start = time.time()

        try:
            while alive_workers or not traj_queue.empty():
                # 1. 推理调度
                inferred = self._dispatch_inference(main_pipes, pending)

                # 2. 收集轨迹
                new_hands = self._collect_trajectories(
                    traj_queue, traj_buffer, results_buffer, alive_workers
                )
                self.total_hands += new_hands

                # 3. PPO 训练
                buf_size = len(traj_buffer.get('states', []))
                if buf_size >= self.ppo_buffer_size:
                    loss = self._do_ppo_train(traj_buffer)
                    if loss is not None:
                        elapsed = time.time() - training_start
                        hps = self.total_hands / max(elapsed, 1e-6)
                        logger.info(
                            f"[SEED RL] hands={self.total_hands}/{num_hands}, "
                            f"PPO更新#{self.total_ppo_updates}, loss={loss:.6f}, "
                            f"速度={hps:.1f} hands/s"
                        )

                # 4. 定期评估
                if (self.total_hands - last_eval_hand) >= self.eval_interval and self.total_hands > 0:
                    last_eval_hand = self.total_hands
                    eval_res = self._run_evaluation(self.num_eval_games)
                    score = eval_res['avg_profit']
                    logger.info(
                        f"[SEED RL] 评估@{self.total_hands}: "
                        f"胜率={eval_res['win_rate']:.3f}, "
                        f"平均收益={score:.2f}, "
                        f"无效下注率={eval_res['invalid_bet_rate']:.3f}"
                    )
                    if score > self.best_score and eval_res['invalid_bet_rate'] < 0.5:
                        self.best_score = score
                        self.save_checkpoint(is_best=True)

                # 5. 定期保存 checkpoint
                if (self.total_hands - last_ckpt_hand) >= self.checkpoint_interval and self.total_hands > 0:
                    last_ckpt_hand = self.total_hands
                    self.save_checkpoint(is_best=False)

                # 6. 定期更新对手模型权重（仅此一次 set_weights，而非每次推理）
                if (self.total_hands - last_opp_update_hand) >= self.opponent_update_interval and self.total_hands > 0:
                    last_opp_update_hand = self.total_hands
                    self.opp_model.set_weights(self.model.get_weights())
                    logger.info(f"[SEED RL] 对手权重已更新 @{self.total_hands}")

                # 短暂 sleep 避免 CPU 空转
                if inferred == 0 and new_hands == 0:
                    time.sleep(0.0001)

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

        # 训练结束后处理剩余 buffer
        remaining = len(traj_buffer.get('states', []))
        if remaining > 0:
            logger.info(f"[SEED RL] 处理剩余 buffer: {remaining} steps")
            # 降低阈值强制训练
            old_buf = self.ppo_buffer_size
            self.ppo_buffer_size = 1
            self._do_ppo_train(traj_buffer)
            self.ppo_buffer_size = old_buf

        elapsed = time.time() - training_start
        hps = self.total_hands / max(elapsed, 1e-6)
        logger.info(
            f"[SEED RL] 训练完成: "
            f"总手牌={self.total_hands}, 总PPO更新={self.total_ppo_updates}, "
            f"耗时={elapsed:.1f}s, 速度={hps:.1f} hands/s"
        )

        # 最终 checkpoint
        self.save_checkpoint(is_best=False)

        # 最终评估
        if not self.skip_final_eval:
            logger.info("[SEED RL] 最终评估...")
            final_eval = self._run_evaluation(self.num_eval_games)
            logger.info(
                f"[SEED RL] 最终评估结果: "
                f"胜率={final_eval['win_rate']:.3f}, "
                f"平均收益={final_eval['avg_profit']:.2f}, "
                f"无效下注率={final_eval['invalid_bet_rate']:.3f}"
            )
