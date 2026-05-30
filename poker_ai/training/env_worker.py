"""SEED RL 环境 Worker 进程 — 纯游戏逻辑，不导入 TensorFlow

每个 worker 是独立进程，运行扑克游戏模拟。
当任何 NN 玩家（主 agent 或 PPO 对手）需要决策时：
  1. 编码状态 → 通过 Pipe 发送给主进程
  2. 主进程在 GPU 上批量推理 → 通过 Pipe 返回 (action_probs, value)
  3. Worker 本地执行合法动作掩码 + 翻牌前位置过滤 → 采样动作 → 继续游戏

Fish 对手直接本地随机决策，不走 GPU。
手牌结束后，通过 traj_queue 回传轨迹数据。
"""

import json
import numpy as np
import os
import random
import traceback
from typing import Dict, List, Tuple, Optional
from multiprocessing import Queue
from multiprocessing.connection import Connection

from ..game.poker_game import PokerGame, Player
from ..model.state_representation import StateRepresentation
from ..reward.reward_calculator import RewardCalculator
from ..action_space import (
    ACTION_DIM,
    ALLIN_ACTION_INDEX,
    POSTFLOP_POT_FRACTIONS,
    PREFLOP_RAISE_MULTIPLIERS,
    RAISE_ACTION_START,
)

# ---- C++ 引擎 (可选) ----
try:
    from ..engine import USE_CC_ENGINE
    if USE_CC_ENGINE:
        from ..engine.poker_engine_cc import GameEngine, decode_action as cc_decode_action
except ImportError:
    USE_CC_ENGINE = False

# 探索保障：按动作类型设置很低的最低采样概率，防止早期策略完全坍缩。
# 扩到 12 维后保底必须更克制，否则会人为推高加注/全下频率。
_ACTION_MIN_PROBS = np.array(
    [0.001, 0.001, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001],
    dtype=np.float32,
)


# ========================================================================
#  合法动作掩码 / 翻牌前位置过滤（复制自 mp_worker.py，避免 TF 导入）
# ========================================================================

_PREFLOP_THRESHOLDS = [0.30, 0.35, 0.25, 0.50, 0.45, 0.35]


def _load_and_convert_preflop_lut():
    """加载 preflop_equity.json 并转换为 C++ 引擎的 key 编码格式"""
    lut_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'preflop_equity.json')
    if not os.path.exists(lut_path):
        return None
    with open(lut_path, 'r') as f:
        raw = json.load(f)

    rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

    converted = {}
    for key_str, equity_dict in raw.items():
        # 解析 key_str
        if len(key_str) == 2:  # pair, e.g. "22", "AA"
            r = rank_map.get(key_str[0])
            if r is None:
                continue
            high, low, suited = r, r, False
        elif len(key_str) == 3:  # e.g. "AKs", "32o"
            r1 = rank_map.get(key_str[0])
            r2 = rank_map.get(key_str[1])
            if r1 is None or r2 is None:
                continue
            high = max(r1, r2)
            low = min(r1, r2)
            suited = (key_str[2] == 's')
        else:
            continue

        # C++ key 编码: (high_rank << 8) | (low_rank << 4) | (suited ? 1 : 0)
        cpp_key = (high << 8) | (low << 4) | (1 if suited else 0)
        converted[cpp_key] = equity_dict

    return converted if converted else None


def _legal_mask(game: PokerGame, player: Player) -> np.ndarray:
    mask = np.zeros(ACTION_DIM, dtype=np.float32)
    if not player.is_active or player.is_all_in:
        mask[0] = 1.0
        return mask
    call_amount = max(0, game.current_bet - player.bet)
    mask[0] = 1.0
    if call_amount <= player.stack:
        mask[1] = 1.0
    cb = max(game.current_bet, game.big_blind)
    min_raise = max(game.current_bet * 2, game.big_blind)
    ai = int(player.bet + player.stack)
    if game.round == 1:  # 翻牌前
        bets = [int(cb * m) for m in PREFLOP_RAISE_MULTIPLIERS]
        for i, bet in enumerate(bets, start=2):
            if bet > game.current_bet and bet >= min_raise and (bet - player.bet) <= player.stack:
                mask[i] = 1.0
    else:  # 翻牌后
        pot = max(game.pot, game.big_blind)
        bets = [int(cb + pot * f) for f in POSTFLOP_POT_FRACTIONS]
        for i, bet in enumerate(bets, start=2):
            if bet > game.current_bet and bet >= min_raise and (bet - player.bet) <= player.stack:
                mask[i] = 1.0
    if ai > game.current_bet and player.stack > 0:
        mask[ALLIN_ACTION_INDEX] = 1.0
    if np.sum(mask) <= 0:
        mask[0] = 1.0
        if call_amount <= player.stack:
            mask[1] = 1.0
    return mask


def _preflop_filter(probs: np.ndarray, game: PokerGame,
                    player: Player, hs: float) -> np.ndarray:
    if game.round != 1:
        return probs
    n = len(game.players)
    dist = (game.players.index(player) - game.dealer_pos + n) % n
    thr = _PREFLOP_THRESHOLDS[min(dist, 5)]
    if hs >= thr:
        return probs
    r = max(0.05, hs / thr)
    a = probs.copy()
    for idx in range(RAISE_ACTION_START, ALLIN_ACTION_INDEX):
        a[idx] *= r
    a[ALLIN_ACTION_INDEX] *= r * 0.1
    s = float(np.sum(a))
    return a / s if s > 1e-8 else probs


def _apply_mask_and_sample(
    action_probs: np.ndarray, game: PokerGame, player: Player,
    sr: StateRepresentation,
) -> Tuple[str, Optional[int], int, float]:
    """对 GPU 返回的 action_probs 执行掩码 + 采样 → (act, amt, idx, old_prob)"""
    lm = _legal_mask(game, player)
    mp_ = action_probs * lm
    s = float(np.sum(mp_))
    mp_ = mp_ / s if s > 1e-8 else lm / (np.sum(lm) + 1e-8)

    # === 探索保障：按动作类型设置最低采样概率 ===
    action_min_probs = _ACTION_MIN_PROBS
    legal_min = action_min_probs * lm  # 非法动作自动为0
    mp_ = np.maximum(mp_, legal_min)   # 一步到位，向量化提升
    mp_ = mp_ / (np.sum(mp_) + 1e-9)  # 重新归一化

    # 移除 _preflop_filter：训练采样时不应干预 action 分布，
    # 否则 old_prob(过滤后) 与 new_prob(未过滤) 产生虚假 ratio，导致 PPO 梯度被污染，
    # 模型学不会加注。与 C++ 路径 (_apply_mask_and_sample_cc) 保持一致。
    hs = sr.calculate_hand_strength(player, game)

    ai = int(np.random.choice(ACTION_DIM, p=mp_))
    av = np.zeros(ACTION_DIM, dtype=np.float32)
    av[ai] = 1.0
    act, amt = sr.decode_action(av, game, player, hs)
    old_prob = float(np.clip(mp_[ai], 1e-8, 1.0))
    return act, amt, ai, old_prob


# ========================================================================
#  消息协议
# ========================================================================

# Worker → Main  (通过 state_pipe)
#   ('infer', worker_id, player_name, state_vector)   请求推理
#
# Main → Worker  (通过 state_pipe)
#   (action_probs_1d, value_scalar)                    推理结果
#
# Worker → Main  (通过 traj_queue)
#   ('traj', result_dict, trajectory_dict)             手牌结束
#   ('done', worker_id)                                Worker 退出

# ========================================================================
#  Fish 对手（本地随机，不走 GPU）
# ========================================================================

def _fish_action(game: PokerGame, player: Player, sr: StateRepresentation):
    lm = _legal_mask(game, player)
    p = lm / (np.sum(lm) + 1e-8)
    ai = int(np.random.choice(ACTION_DIM, p=p))
    av = np.zeros(ACTION_DIM, dtype=np.float32)
    av[ai] = 1.0
    hs = sr.calculate_hand_strength(player, game)
    return sr.decode_action(av, game, player, hs)


# ========================================================================
#  Worker 主循环（向量化环境 — 每个 worker 同时管理多张牌桌）
# ========================================================================

def env_worker_loop(
    worker_id: int,
    state_pipe: Connection,
    traj_queue: Queue,
    game_cfg: dict,
    opp_type_list: List[str],
    total_hands: int,
):
    """
    向量化环境 Worker 进程入口。

    每个 worker 同时管理 tables_per_worker 张独立牌桌，
    批量收集推理请求后一次性发送，大幅提升 GPU batch size。

    Args:
        worker_id: 本 worker 编号
        state_pipe: 双向 Pipe 与主进程通信
        traj_queue: Queue 用于回传完成的轨迹
        game_cfg: 游戏配置
        opp_type_list: 对手类型列表
        total_hands: 本 worker 需要跑的手牌总数
    """
    sr = StateRepresentation()
    rc = RewardCalculator()

    # 打印引擎类型
    import sys
    engine_tag = "C++ Engine" if USE_CC_ENGINE else "Python Engine"
    print(f"[Worker {worker_id}] 使用 {engine_tag}", file=sys.stderr, flush=True)

    sb = game_cfg['small_blind']
    bb = game_cfg['big_blind']
    stack = game_cfg['initial_stack']
    eq_thr = game_cfg.get('allin_eq_threshold', 0.65)
    bonus_cap = game_cfg.get('allin_bonus_cap', 0.80)
    tables_per_worker = game_cfg.get('tables_per_worker', 8)

    hands_done = 0

    # 牌桌状态结构
    class TableState:
        __slots__ = ['game', 'players', 'arecs', 'allin', 'traj', 'finished']
        def __init__(self):
            self.game = None
            self.players = None
            self.arecs = None
            self.allin = None
            self.traj = None
            self.finished = False

    def _init_table(table: TableState):
        """(re)初始化一张牌桌"""
        players = [Player("Main_AI", stack, True)]
        for i in range(5):
            players.append(Player(f"Opponent_{i + 1}", stack, True))
        game = PokerGame(sb, bb, players)
        game.dealer_pos = random.randrange(6)
        game.start_new_hand()
        table.game = game
        table.players = players
        table.arecs = {p.name: [] for p in players}
        table.allin = {'happened': False, 'round': None, 'equity': 0.0}
        table.traj = {'states': [], 'actions': [], 'old_probs': []}
        table.finished = False

    def _settle_table(table: TableState) -> bool:
        """结算一张牌桌，返回 True 表示已发送轨迹"""
        nonlocal hands_done
        game, players, arecs, allin, traj = (
            table.game, table.players, table.arecs, table.allin, table.traj
        )

        alive = [p for p in players if p.is_active]
        if len(alive) == 1 and game.pot > 0:
            game._award_pot(alive[0])

        fp = players[0].stack - stack
        ws = [p.name for p in getattr(game, 'winners', [])]
        if not ws and hasattr(game, 'winner') and game.winner:
            ws = [game.winner.name]
        iw = "Main_AI" in ws

        fr = rc.calculate_reward(
            player=players[0], game=game, action="end",
            action_amount=0, is_winner=iw, profit_loss=fp, done=True,
        )
        ma = arecs.get("Main_AI", [])

        # ---- Anti-Fold-Bot 勇气奖励（降低版：避免奖励淹没真实胜负信号） ----
        total_actions = len(ma)
        fold_count = sum(1 for a, _ in ma if a == 'fold')
        vpip_count = sum(1 for a, _ in ma if a in ('call', 'raise'))

        if total_actions >= 1:
            # 全弃牌且亏钱 → 轻微负反馈
            if fold_count == total_actions and fp <= 0:
                fr -= 0.3
            # 勇气奖励：主动入池 +0.3（原 +2.0 太大，导致无脑 call 策略）
            if vpip_count > 0:
                fr += 0.3
            # 翻牌前单手弃牌 → 微弱惩罚
            if total_actions == 1 and ma[0][0] == 'fold':
                fr -= 0.1

        # all-in 补偿
        ab = 0.0
        if not iw and allin['happened'] and allin['round'] in (1, 2):
            eq = allin['equity']
            if eq >= eq_thr:
                ub = min(bonus_cap, 0.20 + (eq - eq_thr) * 2.0)
                fr += ub
                ab = ub

        traj['final_reward'] = fr
        traj['allin_unlucky'] = {
            'triggered': ab > 0, 'bonus': ab, 'equity': allin['equity'],
        }

        agg = raises = 0
        for a, _ in ma:
            if a in ('raise', 'call'):
                agg += 1
            if a == 'raise':
                raises += 1

        res = {
            'winner': game.winner.name if hasattr(game, 'winner') and game.winner else 'Unknown',
            'winners': ws,
            'hero_position_index': int((0 - game.dealer_pos + len(game.players)) % len(game.players)),
            'player_profits': {p.name: p.stack - stack for p in players},
            'action_histories': arecs,
            'metrics': {'bluffs': 0, 'aggressive': agg, 'raises': raises, 'invalid_bets': 0},
        }

        traj_queue.put(('traj', res, traj))
        hands_done += 1
        table.finished = True
        return True

    def _advance_table_until_infer(table: TableState) -> Optional[Tuple]:
        """
        推进一张牌桌直到需要 NN 推理或手牌结束。
        返回 (player_name, state_vec) 如果需要推理，否则 None。
        """
        game = table.game
        while True:
            if game.is_hand_over():
                _settle_table(table)
                return None

            if game.is_round_over():
                if not game.proceed_to_next_round():
                    _settle_table(table)
                    return None

            cp = game.next_player()
            if not cp or not cp.is_active:
                continue

            pname = cp.name
            is_main = (pname == "Main_AI")

            if is_main:
                state_vec = sr.get_state(game, cp)
                return (pname, state_vec, cp, True)  # 需要推理

            # 对手处理
            oi = int(pname.split('_')[-1]) - 1 if '_' in pname else 0
            oi = max(0, min(oi, len(opp_type_list) - 1))
            opp_kind = opp_type_list[oi]

            if opp_kind == 'fish':
                act, amt = _fish_action(game, cp, sr)
                table.arecs[pname].append((act, amt if amt else 0))
                if not game.handle_action(cp, act, amt):
                    game.handle_action(cp, 'fold', None)
            else:
                # PPO 对手也需要推理
                state_vec = sr.get_state(game, cp)
                return (pname, state_vec, cp, False)  # 需要推理

    def _apply_infer_result(
        table: TableState, cp: Player, pname: str,
        action_probs: np.ndarray, value: float, is_main: bool
    ):
        """将推理结果应用到牌桌。"""
        game = table.game
        act, amt, aidx, old_prob = _apply_mask_and_sample(action_probs, game, cp, sr)
        table.arecs[pname].append((act, amt if amt else 0))
        ok = game.handle_action(cp, act, amt)

        if is_main:
            # 记录轨迹
            state_vec = sr.get_state(game, cp) if not hasattr(cp, '_last_state') else cp._last_state
            # 实际上 state_vec 在 _advance_table_until_infer 中已经编码了
            # 这里使用之前编码的 state
            if ok:
                if not table.allin['happened'] and cp.is_all_in and game.round in (1, 2):
                    try:
                        eq = float(sr.calculate_hand_strength(cp, game))
                    except Exception:
                        eq = 0.0
                    table.allin.update({'happened': True, 'round': int(game.round),
                                        'equity': max(0.0, min(1.0, eq))})

            if not ok:
                game.handle_action(cp, 'fold', None)
                lm = _legal_mask(game, cp)
                fp = float(np.clip(action_probs[0], 1e-8, 1.0)) if np.sum(lm) > 0 else 1.0
                table.traj['actions'].append(0)
                table.traj['old_probs'].append(fp)
            else:
                table.traj['actions'].append(int(aidx))
                table.traj['old_probs'].append(old_prob)
        else:
            if not ok:
                game.handle_action(cp, 'fold', None)

    # ---- C++ 引擎快速路径 ----
    if USE_CC_ENGINE:
        _env_worker_loop_cc(
            worker_id, state_pipe, traj_queue, game_cfg,
            opp_type_list, total_hands, sb, bb, stack,
            eq_thr, bonus_cap, tables_per_worker,
        )
        return

    try:
        # 初始化所有牌桌
        tables = [TableState() for _ in range(tables_per_worker)]
        for t in tables:
            if hands_done < total_hands:
                _init_table(t)
            else:
                t.finished = True

        while hands_done < total_hands:
            # ---- 第 1 步：推进所有牌桌，收集推理请求 ----
            pending_infers = []  # [(table_idx, pname, state_vec, cp, is_main), ...]

            for tidx, table in enumerate(tables):
                if table.finished:
                    # 重启牌桌
                    if hands_done < total_hands:
                        _init_table(table)
                    else:
                        continue

                result = _advance_table_until_infer(table)
                if result is not None:
                    pname, state_vec, cp, is_main = result
                    # 如果是主 agent，记录 state——轨迹用
                    if is_main:
                        table.traj['states'].append(state_vec)
                    pending_infers.append((tidx, pname, state_vec, cp, is_main))

            if not pending_infers:
                # 所有牌桌都结束了
                if all(t.finished for t in tables):
                    break
                continue

            # ---- 第 2 步：批量发送推理请求 ----
            for tidx, pname, state_vec, cp, is_main in pending_infers:
                state_pipe.send(('infer', worker_id, pname, state_vec))

            # ---- 第 3 步：批量接收推理结果（保持顺序）----
            for tidx, pname, state_vec, cp, is_main in pending_infers:
                action_probs, value = state_pipe.recv()
                _apply_infer_result(tables[tidx], cp, pname, action_probs, value, is_main)

    except Exception as e:
        traj_queue.put(('error', worker_id, str(e), traceback.format_exc()))
    finally:
        traj_queue.put(('done', worker_id))


# ========================================================================
#  C++ 引擎快速路径 —— 全部用 index-based API，零 Python 对象分配
# ========================================================================

def _env_worker_loop_cc(
    worker_id: int,
    state_pipe: Connection,
    traj_queue: Queue,
    game_cfg: dict,
    opp_type_list: List[str],
    total_hands: int,
    sb: int, bb: int, stack: int,
    eq_thr: float, bonus_cap: float,
    tables_per_worker: int,
):
    """C++ 引擎 Worker 主循环，完全代替 Python 路径。"""
    MAIN_IDX = 0  # 主 agent 始终是 player 0
    ACTION_MAP = {0: 0, 1: 1, **{i: 2 for i in range(2, ACTION_DIM)}}  # aidx → C++ action
    hands_done = 0

    # ---- 简化的羻牌前位置过滤 ----
    PREFLOP_THR = [0.30, 0.35, 0.25, 0.50, 0.45, 0.35]

    def _preflop_filter_cc(probs, eng, pi, hs):
        if eng.round != 1:
            return probs
        dist = (pi - eng.dealer_pos + eng.num_players) % eng.num_players
        thr = PREFLOP_THR[min(dist, 5)]
        if hs >= thr:
            return probs
        r = max(0.05, hs / thr)
        a = probs.copy()
        for idx in range(RAISE_ACTION_START, ALLIN_ACTION_INDEX):
            a[idx] *= r
        a[ALLIN_ACTION_INDEX] *= r * 0.1
        s = float(np.sum(a))
        return a / s if s > 1e-8 else probs

    def _apply_mask_and_sample_cc(action_probs, eng, pi):
        """C++ 版本的掩码 + 采样"""
        lm = np.array(eng.get_legal_mask(pi), dtype=np.float32)
        mp_ = action_probs * lm
        s = float(np.sum(mp_))
        mp_ = mp_ / s if s > 1e-8 else lm / (np.sum(lm) + 1e-8)

        # === 探索保障：按动作类型设置最低采样概率 ===
        action_min_probs = _ACTION_MIN_PROBS
        legal_min = action_min_probs * lm  # 非法动作自动为0
        mp_ = np.maximum(mp_, legal_min)   # 一步到位，向量化提升
        mp_ = mp_ / (np.sum(mp_) + 1e-9)  # 重新归一化

        hs = eng.get_hand_strength(pi)
        # 移除 _preflop_filter_cc：训练采样时不应干预 action 分布，
        # 否则 old_prob(过滤后) 与 new_prob(未过滤) 产生虚假 ratio，导致 PPO 梯度被污染，模型学不会加注。
        aidx = int(np.random.choice(ACTION_DIM, p=mp_))
        # 解码动作
        act_str, amt = cc_decode_action(
            aidx, eng.current_bet, eng.big_blind,
            eng.get_player_bet(pi), eng.get_player_stack(pi),
            eng.round, eng.pot,
        )
        old_prob = float(np.clip(mp_[aidx], 1e-8, 1.0))
        return act_str, amt, aidx, old_prob, hs

    def _fish_action_cc(eng, pi):
        lm = np.array(eng.get_legal_mask(pi), dtype=np.float32)
        p = lm / (np.sum(lm) + 1e-8)
        aidx = int(np.random.choice(ACTION_DIM, p=p))
        act_str, amt = cc_decode_action(
            aidx, eng.current_bet, eng.big_blind,
            eng.get_player_bet(pi), eng.get_player_stack(pi),
            eng.round, eng.pot,
        )
        return act_str, amt, aidx

    # ---- C++ 牌桌状态 ----
    class CCTable:
        __slots__ = ['eng', 'arecs', 'allin', 'traj', 'finished']
        def __init__(self):
            self.eng = None
            self.arecs = None
            self.allin = None
            self.traj = None
            self.finished = False

    # 进程级缓存：在 _env_worker_loop_cc 内部加载一次，避免每张牌桌重复读取
    _preflop_lut_for_cc = _load_and_convert_preflop_lut()

    def _init_cc(table: CCTable):
        eng = GameEngine(sb, bb, stack, 6)
        eng.start_new_hand(random.randrange(6))
        # 传入 preflop equity LUT，与 Python 端查表逻辑对齐
        if _preflop_lut_for_cc is not None:
            eng.set_preflop_lut(_preflop_lut_for_cc)
        table.eng = eng
        table.arecs = {i: [] for i in range(6)}
        table.allin = {'happened': False, 'round': None, 'equity': 0.0}
        table.traj = {'states': [], 'actions': [], 'old_probs': []}
        table.finished = False

    def _settle_cc(table: CCTable) -> bool:
        nonlocal hands_done
        eng = table.eng
        arecs = table.arecs
        allin = table.allin
        traj = table.traj

        fp = eng.get_player_stack(MAIN_IDX) - stack
        w_indices = list(eng.get_winner_indices())
        iw = MAIN_IDX in w_indices

        # 简化奖励计算（避免创建 Python Player/Game 代理对象）
        fr = float(np.clip(fp / max(bb, 1), -2.0, 2.0))
        if iw:
            fr += 0.5

        # ---- Anti-Fold-Bot 勇气奖励（降低版） ----
        ma = arecs.get(MAIN_IDX, [])
        total_actions = len(ma)
        fold_count = sum(1 for a, _ in ma if a == 'fold')
        vpip_count = sum(1 for a, _ in ma if a in ('call', 'raise'))

        if total_actions >= 1:
            if fold_count == total_actions and fp <= 0:
                fr -= 0.3
            if vpip_count > 0:
                fr += 0.3
            if total_actions == 1 and ma[0][0] == 'fold':
                fr -= 0.1

        # 加注专项奖励：防止模型永远 call，鼓励探索 raise
        raise_count = sum(1 for a, _ in ma if a == 'raise')
        if raise_count > 0:
            fr += 0.1  # 加注动作额外 +0.1（降低远 +0.2，避免初期 allin 爆炸）

        # all-in 补偿
        ab = 0.0
        if not iw and allin['happened'] and allin['round'] in (1, 2):
            eq = allin['equity']
            if eq >= eq_thr:
                ub = min(bonus_cap, 0.20 + (eq - eq_thr) * 2.0)
                fr += ub
                ab = ub

        fr = float(np.clip(fr, -2.0, 3.0))  # 上限从 2.0 扩大到 3.0，确保 raise+win 奖励可超过纯 call+win
        traj['final_reward'] = fr
        traj['allin_unlucky'] = {
            'triggered': ab > 0, 'bonus': ab, 'equity': allin['equity'],
        }

        # 统计指标
        agg = raises = 0
        for a, _ in ma:
            if a in ('raise', 'call'):
                agg += 1
            if a == 'raise':
                raises += 1

        ws_names = [f"Player_{w}" for w in w_indices]
        profits = {f"Player_{i}": eng.get_player_stack(i) - stack for i in range(6)}
        arecs_named = {}
        for i in range(6):
            name = "Main_AI" if i == 0 else f"Opponent_{i}"
            arecs_named[name] = arecs.get(i, [])

        res = {
            'winner': ws_names[0] if ws_names else 'Unknown',
            'winners': ws_names,
            'hero_position_index': int((0 - eng.dealer_pos + eng.num_players) % eng.num_players),
            'player_profits': profits,
            'action_histories': arecs_named,
            'metrics': {'bluffs': 0, 'aggressive': agg, 'raises': raises, 'invalid_bets': 0},
        }
        traj_queue.put(('traj', res, traj))
        hands_done += 1
        table.finished = True
        return True

    def _advance_cc(table: CCTable):
        """推进 C++ 牌桌直到需要 NN 推理或手牌结束。"""
        eng = table.eng
        while True:
            if eng.is_hand_over():
                _settle_cc(table)
                return None

            if eng.is_round_over():
                done = eng.proceed_to_next_round()
                if done:
                    _settle_cc(table)
                    return None

            pi = eng.next_player()
            if pi < 0:
                # 没有可行动的玩家，尝试推进到下一轮
                if eng.is_round_over():
                    done = eng.proceed_to_next_round()
                    if done:
                        _settle_cc(table)
                        return None
                    continue
                _settle_cc(table)
                return None

            is_main = (pi == MAIN_IDX)

            if is_main:
                state_vec = np.array(eng.get_state_vector(pi), dtype=np.float32)
                return (pi, state_vec, is_main)

            # 对手处理
            oi = max(0, min(pi - 1, len(opp_type_list) - 1))
            opp_kind = opp_type_list[oi]

            if opp_kind == 'fish':
                act_str, amt, aidx = _fish_action_cc(eng, pi)
                table.arecs[pi].append((act_str, amt if amt else 0))
                cc_act = ACTION_MAP[aidx]
                if not eng.handle_action(pi, cc_act, amt if amt else 0):
                    eng.handle_action(pi, 0, 0)  # fold
            else:
                # PPO 对手也需要推理
                state_vec = np.array(eng.get_state_vector(pi), dtype=np.float32)
                return (pi, state_vec, is_main)

    def _apply_cc_result(table: CCTable, pi: int, is_main: bool,
                         action_probs: np.ndarray, value: float):
        eng = table.eng
        act_str, amt, aidx, old_prob, hs = _apply_mask_and_sample_cc(action_probs, eng, pi)
        table.arecs[pi].append((act_str, amt if amt else 0))

        cc_act = ACTION_MAP[aidx]
        ok = eng.handle_action(pi, cc_act, amt if amt else 0)

        if is_main:
            if ok:
                if not table.allin['happened'] and eng.get_player_is_all_in(pi) and eng.round in (1, 2):
                    try:
                        eq = float(eng.get_hand_strength(pi))
                    except Exception:
                        eq = 0.0
                    table.allin.update({'happened': True, 'round': int(eng.round),
                                        'equity': max(0.0, min(1.0, eq))})

            if not ok:
                eng.handle_action(pi, 0, 0)  # fold
                lm = np.array(eng.get_legal_mask(pi), dtype=np.float32)
                fp = float(np.clip(action_probs[0], 1e-8, 1.0)) if np.sum(lm) > 0 else 1.0
                table.traj['actions'].append(0)
                table.traj['old_probs'].append(fp)
            else:
                table.traj['actions'].append(int(aidx))
                table.traj['old_probs'].append(old_prob)
        else:
            if not ok:
                eng.handle_action(pi, 0, 0)  # fold

    try:
        tables = [CCTable() for _ in range(tables_per_worker)]
        for t in tables:
            if hands_done < total_hands:
                _init_cc(t)
            else:
                t.finished = True

        while hands_done < total_hands:
            pending = []
            for tidx, table in enumerate(tables):
                if table.finished:
                    if hands_done < total_hands:
                        _init_cc(table)
                    else:
                        continue
                result = _advance_cc(table)
                if result is not None:
                    pi, state_vec, is_main = result
                    if is_main:
                        table.traj['states'].append(state_vec)
                    pname = "Main_AI" if pi == MAIN_IDX else f"Opponent_{pi}"
                    pending.append((tidx, pname, state_vec, pi, is_main))

            if not pending:
                if all(t.finished for t in tables):
                    break
                continue

            for tidx, pname, state_vec, pi, is_main in pending:
                state_pipe.send(('infer', worker_id, pname, state_vec))

            for tidx, pname, state_vec, pi, is_main in pending:
                action_probs, value = state_pipe.recv()
                _apply_cc_result(tables[tidx], pi, is_main, action_probs, value)

    except Exception as e:
        traj_queue.put(('error', worker_id, str(e), traceback.format_exc()))
    finally:
        traj_queue.put(('done', worker_id))
