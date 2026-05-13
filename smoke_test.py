#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
冒烟测试脚本 (smoke test)

目的：在服务器上首次部署后用最快方式验证：
  1) 依赖已装好（TF / numpy / matplotlib / 项目内模块都能 import）
  2) GPU 是否被 TensorFlow 识别（识别不到只警告，不报错）
  3) 6-Max 游戏循环能跑通一局
  4) PPO / DQN 三种 agent 都能完成一次 select_action（前向推理）
  5) SelfPlayTrainer 能创建 + 完成一次 evaluate(num_eval_games=2)

不进行任何训练，不写检查点，运行时间预期 < 60s（取决于 TF 启动）。

退出码：
  0  全部通过
  1  关键检查失败
"""
import os
import sys
import time
import traceback


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def main() -> int:
    failures = []

    # 1. 基础依赖
    section("1. 基础依赖检查")
    try:
        import numpy as np
        import matplotlib  # noqa: F401
        print(f"  numpy   : {np.__version__}")
        print(f"  python  : {sys.version.split()[0]}")
    except Exception as e:
        failures.append(f"基础依赖 import 失败: {e}")
        traceback.print_exc()

    # 2. TensorFlow 与 GPU
    section("2. TensorFlow / GPU 检查")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        print(f"  TensorFlow 版本: {tf.__version__}")
        print(f"  可用 GPU 数量  : {len(gpus)}")
        for i, g in enumerate(gpus):
            print(f"    GPU[{i}] = {g.name}")
        if gpus:
            for g in gpus:
                try:
                    tf.config.experimental.set_memory_growth(g, True)
                except Exception as e:
                    print(f"  [WARN] set_memory_growth 失败 ({g.name}): {e}")
            with tf.device('/GPU:0'):
                a = tf.random.normal([1024, 1024])
                b = tf.random.normal([1024, 1024])
                c = tf.matmul(a, b)
                _ = float(c[0, 0].numpy())
            print(f"  GPU 矩阵乘法 OK，结果设备: {c.device}")
        else:
            print("  [INFO] 未检测到 GPU，将使用 CPU（训练会慢，但仍可运行）")
    except Exception as e:
        failures.append(f"TensorFlow 初始化失败: {e}")
        traceback.print_exc()

    # 3. 项目模块 import
    section("3. 项目模块 import 检查")
    try:
        from poker_ai.game.poker_game import PokerGame, Player
        from poker_ai.game.cards import Deck, Card
        from poker_ai.model.ai_agent import PokerAIAgent
        from poker_ai.model.hand_evaluator import HandEvaluator
        from poker_ai.advisor.ai_advisor import AIAvisor, AIAdvisor
        from poker_ai.reward.self_play_trainer import SelfPlayTrainer
        assert AIAvisor is AIAdvisor, "AIAdvisor 别名应等于 AIAvisor"
        print("  全部业务模块 import 成功，AIAdvisor 别名 OK")
    except Exception as e:
        failures.append(f"业务模块 import 失败: {e}")
        traceback.print_exc()
        # 这里如果 import 都失败，后面的测试也没法做
        return 1

    # 4. 单局 6-Max 牌局推进
    section("4. 单局 6-Max 牌局")
    try:
        players = [Player(f"P{i+1}", 1000, True) for i in range(6)]
        game = PokerGame(5, 10, players)
        game.start_new_hand()
        # 简单跑两轮，每轮所有玩家都跟注或弃牌
        for _ in range(20):
            if game.is_round_over():
                if game.proceed_to_next_round():
                    break
                continue
            p = game.next_player()
            if p is None:
                break
            game.handle_action(p, "call")
        print(f"  单局推进 OK，最终底池 = {game.pot}")
    except Exception as e:
        failures.append(f"6-Max 牌局推进失败: {e}")
        traceback.print_exc()

    # 5. 三种 agent 各做一次 select_action
    section("5. 三种 agent 推理")
    for agent_type in ("dqn", "policy_gradient", "ppo"):
        try:
            agent = PokerAIAgent(agent_type=agent_type)
            players = [Player(f"P{i+1}", 1000, True) for i in range(6)]
            g = PokerGame(5, 10, players)
            g.start_new_hand()
            action, amount = agent.select_action(g, players[0])
            print(f"  [{agent_type}] action={action} amount={amount}")
        except Exception as e:
            failures.append(f"agent={agent_type} 推理失败: {e}")
            traceback.print_exc()

    # 6. SelfPlayTrainer.evaluate(2 局)
    section("6. SelfPlayTrainer evaluate (2 局)")
    try:
        t0 = time.perf_counter()
        trainer = SelfPlayTrainer(
            model_type="ppo",
            checkpoint_dir="./_smoke_ckpt",
            num_eval_games=2,
            eval_workers=1,
            train_workers=1,
        )
        result = trainer.evaluate()
        elapsed = time.perf_counter() - t0
        print(f"  evaluate() OK，耗时 {elapsed:.2f}s，结果: {result}")
    except Exception as e:
        failures.append(f"SelfPlayTrainer.evaluate 失败: {e}")
        traceback.print_exc()

    # 总结
    section("总结")
    if failures:
        print(f"  失败 {len(failures)} 项:")
        for i, f in enumerate(failures, 1):
            print(f"    {i}. {f}")
        return 1
    print("  全部冒烟测试通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())