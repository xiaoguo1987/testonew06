#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PPO模型简单测试脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from poker_ai.model.ai_agent import PokerAIAgent

print("测试PPO模型初始化...")

try:
    # 创建PPO代理
    ppo_agent = PokerAIAgent(agent_type='ppo')
    print("✅ PPO代理创建成功")
    
    # 测试动作选择
    from poker_ai.game.poker_game import PokerGame, Player
    
    # 创建测试游戏环境
    players = [Player("TestPlayer", 1000, True) for _ in range(6)]
    game = PokerGame(5, 10, players)
    game.start_new_hand()
    
    print("✅ 游戏环境创建成功")
    
    # 测试动作选择
    action, amount = ppo_agent.select_action(game, players[0])
    print(f"✅ 动作选择成功: {action}, {amount}")
    
    print("\n🎉 所有测试通过！PPO模型可以正常工作")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
