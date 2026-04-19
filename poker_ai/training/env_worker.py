"""SEED RL 环境 Worker 进程 — 纯游戏逻辑，不导入 TensorFlow

每个 worker 是独立进程，运行扑克游戏模拟。
当任何 NN 玩家（主 agent 或 PPO 对手）需要决策时：
  1. 编码状态 → 通过 Pipe 发送给主进程
  2. 主进程在 GPU 上批量推理 → 通过 Pipe 返回 (action_probs, value)
  3. Worker 本地执行合法动作掩码 + 翻牌前位置过滤 → 采样动作 → 继续游戏

Fish 对手直接本地随机决策，不走 GPU。
手牌结束后，通过 traj_queue 回传轨迹数据。
"""

import numpy as np
import random
import traceback
from typing import Dict, List, Tuple, Optional
from multiprocessing import Queue
from multiprocessing.connection import Connection

from ..game.poker_game import PokerGame, Player
from ..model.state_representation import StateRepresentation
from ..reward.reward_calculator import RewardCalculator


# ========================================================================
#  合法动作掩码 / 翻牌前位置过滤（复制自 mp_worker.py，避免 TF 导入）
# ========================================================================

_PREFLOP_THRESHOLDS = [0.30, 0.35, 0.25, 0.50, 0.45, 0.35]


def _legal_mask(game: PokerGame, player: Player) -> np.ndarray:
    mask = np.zeros(5, dtype=np.float32)
    if not player.is_active or player.is_all_in:
        mask[0] = 1.0
        return mask
    call_amount = max(0, game.current_bet - player.bet)
    mask[0] = 1.0
    if call_amount <= player.stack:
        mask[1] = 1.0
    cb = max(game.current_bet, game.big_blind)
    r2, r4 = int(cb * 2), int(cb * 4)
    ai = int(player.bet + player.stack)
    mr = max(game.current_bet * 2, game.big_blind)
    if r2 > game.current_bet and r2 >= mr and (r2 - player.bet) <= player.stack:
        mask[2] = 1.0
    if r4 > game.current_bet and r4 >= mr and (r4 - player.bet) <= player.stack:
        mask[3] = 1.0
    if ai > game.current_bet and ai >= mr and player.stack > 0:
        mask[4] = 1.0
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
    a[2] *= r
    a[3] *= r * 0.5
    a[4] *= r * 0.2
    s = float(np.sum(a))
    return a / s if s > 1e-8 else probs


def _apply_mask_and_sample(
    action_probs: np.ndarray, game: PokerGame, player: Player,
    sr: StateRepresentation,
) -> Tuple[str, Optional[int], int, float]:
    """对 GPU 返回的 action_probs 执行掩码 + 过滤 + 采样 → (act, amt, idx, old_prob)"""
    lm = _legal_mask(game, player)
    mp_ = action_probs * lm
    s = float(np.sum(mp_))
    mp_ = mp_ / s if s > 1e-8 else lm / (np.sum(lm) + 1e-8)

    hs = sr.calculate_hand_strength(player, game)
    mp_ = _preflop_filter(mp_, game, player, hs)

    ai = int(np.random.choice(5, p=mp_))
    av = np.zeros(5, dtype=np.float32)
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
    ai = int(np.random.choice(5, p=p))
    av = np.zeros(5, dtype=np.float32)
    av[ai] = 1.0
    hs = sr.calculate_hand_strength(player, game)
    return sr.decode_action(av, game, player, hs)


# ========================================================================
#  Worker 主循环
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
    Worker 进程入口。

    Args:
        worker_id: 本 worker 编号
        state_pipe: 双向 Pipe 与主进程通信（发送推理请求 / 接收推理结果）
        traj_queue: Queue 用于回传完成的轨迹
        game_cfg: {'small_blind', 'big_blind', 'initial_stack',
                    'allin_eq_threshold', 'allin_bonus_cap'}
        opp_type_list: 长度 5 的列表，每个元素 'ppo' 或 'fish'
        total_hands: 本 worker 需要跑的手牌数
    """
    sr = StateRepresentation()
    rc = RewardCalculator()

    sb = game_cfg['small_blind']
    bb = game_cfg['big_blind']
    stack = game_cfg['initial_stack']
    eq_thr = game_cfg.get('allin_eq_threshold', 0.65)
    bonus_cap = game_cfg.get('allin_bonus_cap', 0.80)

    hands_done = 0

    try:
        while hands_done < total_hands:
            # ---- 创建游戏 ----
            players = [Player("Main_AI", stack, True)]
            for i in range(5):
                players.append(Player(f"Opponent_{i + 1}", stack, True))
            game = PokerGame(sb, bb, players)
            game.dealer_pos = random.randrange(6)
            game.start_new_hand()

            arecs: Dict[str, list] = {p.name: [] for p in players}
            allin = {'happened': False, 'round': None, 'equity': 0.0}
            traj: Dict = {'states': [], 'actions': [], 'old_probs': []}

            # ---- 牌局模拟 ----
            while not game.is_hand_over():
                if game.is_round_over():
                    if not game.proceed_to_next_round():
                        break
                cp = game.next_player()
                if not cp or not cp.is_active:
                    continue

                pname = cp.name
                is_main = (pname == "Main_AI")

                if is_main:
                    # 主 Agent：必须走 GPU 推理
                    state_vec = sr.get_state(game, cp)
                    state_pipe.send(('infer', worker_id, pname, state_vec))
                    action_probs, value = state_pipe.recv()  # 阻塞等 GPU 返回

                    act, amt, aidx, old_prob = _apply_mask_and_sample(
                        action_probs, game, cp, sr
                    )
                    arecs[pname].append((act, amt if amt else 0))
                    ok = game.handle_action(cp, act, amt)

                    # all-in 记录
                    if ok and not allin['happened'] and cp.is_all_in and game.round in (1, 2):
                        try:
                            eq = float(sr.calculate_hand_strength(cp, game))
                        except Exception:
                            eq = 0.0
                        allin.update({'happened': True, 'round': int(game.round),
                                      'equity': max(0.0, min(1.0, eq))})

                    if not ok:
                        game.handle_action(cp, 'fold', None)
                        lm = _legal_mask(game, cp)
                        fp = float(np.clip(action_probs[0], 1e-8, 1.0)) if np.sum(lm) > 0 else 1.0
                        traj['states'].append(state_vec)
                        traj['actions'].append(0)
                        traj['old_probs'].append(fp)
                    else:
                        traj['states'].append(state_vec)
                        traj['actions'].append(int(aidx))
                        traj['old_probs'].append(old_prob)
                else:
                    # ---- 对手 ----
                    oi = int(pname.split('_')[-1]) - 1 if '_' in pname else 0
                    oi = max(0, min(oi, len(opp_type_list) - 1))
                    opp_kind = opp_type_list[oi]

                    if opp_kind == 'fish':
                        act, amt = _fish_action(game, cp, sr)
                    else:
                        # PPO 对手：也走 GPU 推理
                        state_vec = sr.get_state(game, cp)
                        state_pipe.send(('infer', worker_id, pname, state_vec))
                        action_probs, _ = state_pipe.recv()
                        act, amt, _, _ = _apply_mask_and_sample(
                            action_probs, game, cp, sr
                        )

                    arecs[pname].append((act, amt if amt else 0))
                    if not game.handle_action(cp, act, amt):
                        game.handle_action(cp, 'fold', None)

            # ---- 结算 ----
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
            if len(ma) == 1 and ma[0][0] == "fold" and fp <= 0:
                fr -= 0.25

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

            # 轻量指标
            agg = raises = 0
            for a, _ in ma:
                if a in ('raise', 'call'):
                    agg += 1
                if a == 'raise':
                    raises += 1

            res = {
                'winner': game.winner.name if hasattr(game, 'winner') and game.winner else 'Unknown',
                'winners': ws,
                'player_profits': {p.name: p.stack - stack for p in players},
                'action_histories': arecs,
                'metrics': {'bluffs': 0, 'aggressive': agg, 'raises': raises, 'invalid_bets': 0},
            }

            traj_queue.put(('traj', res, traj))
            hands_done += 1

    except Exception as e:
        traj_queue.put(('error', worker_id, str(e), traceback.format_exc()))
    finally:
        traj_queue.put(('done', worker_id))
