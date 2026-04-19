#!/usr/bin/env python3
"""生成翻牌前胜率查找表 (Preflop Equity Look-up Table)

169种非等价起手牌 × 1~5个对手 = 845个胜率值
使用蒙特卡洛模拟计算，一次生成，运行时 O(1) 查表。
"""

import json
import os
import sys
import random
from itertools import combinations
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from poker_ai.game.cards import Card, RANKS, SUITS, evaluate_hand

MC_ITERATIONS = 1000  # 每种手牌每个对手数的模拟次数


def _best_hand_rank(seven_cards):
    """从7张牌中选最佳5张组合的牌型"""
    best = None
    for combo in combinations(seven_cards, 5):
        rank = evaluate_hand(list(combo))
        if best is None or rank > best:
            best = rank
    return best


def _calc_equity(args):
    """计算单个手牌类型 vs 1~5 个对手的胜率"""
    hand_key, r1, s1, r2, s2 = args
    hole = [Card(r1, s1), Card(r2, s2)]
    known = {(r1, s1), (r2, s2)}
    deck_template = [Card(r, s) for r in RANKS for s in SUITS if (r, s) not in known]

    equities = {}
    for num_opp in range(1, 6):
        wins = 0
        ties = 0
        for _ in range(MC_ITERATIONS):
            deck = deck_template.copy()
            random.shuffle(deck)
            community = deck[:5]
            my_rank = _best_hand_rank(hole + community)

            idx = 5
            won = True
            tied = False
            for _ in range(num_opp):
                opp_rank = _best_hand_rank(deck[idx:idx + 2] + community)
                if opp_rank > my_rank:
                    won = False
                    tied = False
                    break
                elif opp_rank == my_rank:
                    tied = True
                idx += 2
            if won and not tied:
                wins += 1
            elif won and tied:
                ties += 1

        # 胜率 = 赢的 + 平局的一半
        equities[str(num_opp)] = round((wins + ties * 0.5) / MC_ITERATIONS, 4)

    return hand_key, equities


def main():
    # 生成169种手牌类型的任务参数
    tasks = []
    rank_list = list(reversed(RANKS))  # A, K, Q, ..., 2

    for i, r1 in enumerate(rank_list):
        for j in range(i, len(rank_list)):
            r2 = rank_list[j]
            if i == j:
                # 对子 - 用两种不同花色
                key = f"{r1}{r2}"
                tasks.append((key, r1, SUITS[0], r2, SUITS[1]))
            else:
                # 同花
                key_s = f"{r1}{r2}s"
                tasks.append((key_s, r1, SUITS[0], r2, SUITS[0]))
                # 非同花
                key_o = f"{r1}{r2}o"
                tasks.append((key_o, r1, SUITS[0], r2, SUITS[1]))

    print(f"生成 {len(tasks)} 种手牌类型的胜率 (MC={MC_ITERATIONS})")

    # 多进程并行计算
    num_workers = max(1, cpu_count() - 1)
    print(f"使用 {num_workers} 个进程并行计算")

    result = {}
    with Pool(num_workers) as pool:
        for i, (key, eq) in enumerate(pool.imap_unordered(_calc_equity, tasks)):
            result[key] = eq
            print(f"\r进度: {i + 1}/{len(tasks)} ({key}: vs1={eq['1']}, vs5={eq['5']})", end="", flush=True)

    print(f"\n完成! 生成了 {len(result)} 种手牌类型")

    # 保存到 poker_ai/data/preflop_equity.json
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'poker_ai', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'preflop_equity.json')
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"已保存到: {output_path}")


if __name__ == '__main__':
    main()
