"""多进程采样Worker — 纯NumPy推理，绕过GIL瓶颈

设计要点
1. 不在worker中调用TensorFlow，避免GPU竞争
2. 纯NumPy复现PPO网络前向传播（Dense + LayerNorm + Residual + Softmax）
3. 每个worker进程独立运行游戏模拟+推理，真正利用多核CPU
4. 通过 multiprocessing.Pool 与主进程通信，传递轨迹数据
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional

from ..game.poker_game import PokerGame, Player
from ..model.state_representation import StateRepresentation
from ..reward.reward_calculator import RewardCalculator


# ========================================================================
#  纯 NumPy 前向传播（PPO Actor/Critic 共用架构）
# ========================================================================

def _layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def _forward(x: np.ndarray, weights: list, softmax_output: bool = True) -> np.ndarray:
    """PPO 512-512(残差)-256-128 网络前向传播

    weights 为 model.get_weights() 返回的列表（共 18 个 ndarray）:
      [0,1]   Dense1 kernel/bias       (input→512)
      [2,3]   LayerNorm1 gamma/beta
      [4,5]   Dense2 kernel/bias       (512→512, 残差分支)
      [6,7]   LayerNorm2 gamma/beta
      [8,9]   Dense3 kernel/bias       (512→256)
      [10,11] LayerNorm3 gamma/beta
      [12,13] Dense4 kernel/bias       (256→128)
      [14,15] LayerNorm4 gamma/beta
      [16,17] Output kernel/bias       (128→5 actor / 128→1 critic)
    """
    # Layer 1: Dense(512) → LN → ReLU
    x = x @ weights[0] + weights[1]
    x = _layer_norm(x, weights[2], weights[3])
    x = np.maximum(x, 0)

    # Layer 2: Dense(512) → LN → +残差 → ReLU
    res = x
    x = x @ weights[4] + weights[5]
    x = _layer_norm(x, weights[6], weights[7])
    x = res + x
    x = np.maximum(x, 0)

    # Layer 3: Dense(256) → LN → ReLU
    x = x @ weights[8] + weights[9]
    x = _layer_norm(x, weights[10], weights[11])
    x = np.maximum(x, 0)

    # Layer 4: Dense(128) → LN → ReLU
    x = x @ weights[12] + weights[13]
    x = _layer_norm(x, weights[14], weights[15])
    x = np.maximum(x, 0)

    # Output
    x = x @ weights[16] + weights[17]
    if softmax_output:
        x = _softmax(x)
    return x


# ========================================================================
#  合法动作掩码 / 翻牌前位置过滤（从 ai_agent.py 复制，避免触发 TF 导入）
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


# ========================================================================
#  NumPy PPO Agent / 随机鱼玩家
# ========================================================================

class NumpyPPOAgent:
    """纯 NumPy 推理的 PPO 代理，供多进程 worker 使用。"""
    __slots__ = ('actor_w', 'critic_w', 'sr')

    def __init__(self, actor_w: list, critic_w: list):
        self.actor_w = actor_w
        self.critic_w = critic_w
        self.sr = StateRepresentation()

    def predict(self, state: np.ndarray):
        s = np.atleast_2d(np.asarray(state, dtype=np.float32))
        ap = _forward(s, self.actor_w, softmax_output=True)
        v = _forward(s, self.critic_w, softmax_output=False)
        return ap[0], float(v[0, 0])

    def select_action_with_info(self, game: PokerGame, player: Player):
        state = self.sr.get_state(game, player)
        ap, val = self.predict(state)
        lm = _legal_mask(game, player)
        mp_ = ap * lm
        s = float(np.sum(mp_))
        mp_ = mp_ / s if s > 1e-8 else lm / (np.sum(lm) + 1e-8)
        hs = self.sr.calculate_hand_strength(player, game)
        mp_ = _preflop_filter(mp_, game, player, hs)
        ai = int(np.random.choice(5, p=mp_))
        av = np.zeros(5, dtype=np.float32)
        av[ai] = 1.0
        act, amt = self.sr.decode_action(av, game, player, hs)
        return act, amt, {
            'state': state,
            'action_idx': ai,
            'old_prob': float(np.clip(mp_[ai], 1e-8, 1.0)),
            'action_probabilities': mp_,
            'state_value': val,
        }

    def select_action(self, game: PokerGame, player: Player):
        a, amt, _ = self.select_action_with_info(game, player)
        return a, amt


class RandomFishAgent:
    """全随机代理，等价于 epsilon=1.0 的 DQN。"""
    __slots__ = ('sr',)

    def __init__(self):
        self.sr = StateRepresentation()

    def select_action(self, game: PokerGame, player: Player):
        lm = _legal_mask(game, player)
        p = lm / (np.sum(lm) + 1e-8)
        ai = int(np.random.choice(5, p=p))
        av = np.zeros(5, dtype=np.float32)
        av[ai] = 1.0
        hs = self.sr.calculate_hand_strength(player, game)
        return self.sr.decode_action(av, game, player, hs)


# ========================================================================
#  Worker 进程状态（Pool initializer 写入，mp_run_hands 读取）
# ========================================================================

_W: Dict = {}


def mp_init_worker(game_cfg: dict, reward_cfg: dict,
                   opp_weights_list: list) -> None:
    """multiprocessing.Pool 的 initializer：每个 worker 进程调用一次。"""
    _W['cfg'] = game_cfg
    _W['rcfg'] = reward_cfg
    _W['rc'] = RewardCalculator()
    agents: list = []
    for w in opp_weights_list:
        if w is None:
            agents.append(RandomFishAgent())
        else:
            agents.append(NumpyPPOAgent(w[0], w[1]))
    _W['opp'] = agents


# ========================================================================
#  Worker 核心函数（Pool.map 调用）
# ========================================================================

def mp_run_hands(args: tuple) -> List[Tuple[Dict, Dict]]:
    """在 worker 进程中执行 num_hands 手牌并返回 (result, trajectory) 列表。

    args = (num_hands, main_actor_weights, main_critic_weights)
    """
    num_hands, main_actor_w, main_critic_w = args

    cfg = _W['cfg']
    rcfg = _W['rcfg']
    rc = _W['rc']
    opp_agents = _W['opp']

    main = NumpyPPOAgent(main_actor_w, main_critic_w)
    sb, bb, stack = cfg['small_blind'], cfg['big_blind'], cfg['initial_stack']
    eq_thr = rcfg['allin_eq_threshold']
    bonus_cap = rcfg['allin_bonus_cap']

    out: List[Tuple[Dict, Dict]] = []

    for _ in range(num_hands):
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

            if cp.name == "Main_AI":
                astr, amt, info = main.select_action_with_info(game, cp)
                aidx = info['action_idx']
                cs = info['state']
                arecs[cp.name].append((astr, amt if amt else 0))
                ok = game.handle_action(cp, astr, amt)
                # all-in 记录
                if ok and not allin['happened'] and cp.is_all_in and game.round in (1, 2):
                    try:
                        eq = float(main.sr.calculate_hand_strength(cp, game))
                    except Exception:
                        eq = 0.0
                    allin.update({'happened': True, 'round': int(game.round),
                                  'equity': max(0.0, min(1.0, eq))})
                # 轨迹记录
                if not ok:
                    game.handle_action(cp, 'fold', None)
                    prbs = info.get('action_probabilities')
                    fp = float(np.clip(prbs[0], 1e-8, 1.0)) if prbs is not None else 1.0
                    traj['states'].append(cs)
                    traj['actions'].append(0)
                    traj['old_probs'].append(fp)
                else:
                    traj['states'].append(cs)
                    traj['actions'].append(int(aidx))
                    traj['old_probs'].append(float(info['old_prob']))
            else:
                # ---- 对手 ----
                try:
                    oi = int(cp.name.split('_')[-1]) - 1
                    ag = opp_agents[oi] if 0 <= oi < len(opp_agents) else opp_agents[0]
                except Exception:
                    ag = opp_agents[0]
                act, amt = ag.select_action(game, cp)
                arecs[cp.name].append((act, amt if amt else 0))
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

        # ---- 轻量指标 ----
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
        out.append((res, traj))

    return out
