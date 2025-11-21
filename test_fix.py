#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试德州扑克AI修复方案
该脚本用于验证我们的修复措施是否有效解决了AI累计利润为负的问题
"""

import os
import sys
import logging
import numpy as np
import time
from poker_ai.game.poker_game import Player, PokerGame
from poker_ai.model.ai_agent import PokerAIAgent
from poker_ai.reward.reward_calculator import RewardCalculator

# 设置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("test_fix.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger('TestFix')

def setup_test_environment():
    """设置测试环境"""
    # 创建5个AI玩家和1个人类玩家（用于对比）
    players = [
        Player("Human", 10000, False),  # 人类玩家作为基准
        Player("AI_Fixed_1", 10000, True),  # 修复后的AI玩家1
        Player("AI_Fixed_2", 10000, True),  # 修复后的AI玩家2
        Player("AI_Fixed_3", 10000, True),  # 修复后的AI玩家3
        Player("AI_Fixed_4", 10000, True),  # 修复后的AI玩家4
        Player("AI_Fixed_5", 10000, True)   # 修复后的AI玩家5
    ]
    
    # 初始化每个玩家的累计利润跟踪
    for player in players:
        player.total_profit = 0
    
    # 创建游戏实例，小盲注5，大盲注10
    game = PokerGame(5, 10, players)
    
    # 创建AI代理（使用DQN模型）
    ai_agents = {}
    for player in players:
        if player.is_ai:
            ai_agents[player.name] = PokerAIAgent(agent_type='dqn')
    
    # 创建奖励计算器
    reward_calculator = RewardCalculator()
    
    # 调整奖励权重以测试我们的修复方案
    reward_calculator.reward_weights['aggressive_play'] = 0.05  # 降低进攻性权重
    reward_calculator.reward_weights['bluff_success'] = 0.6    # 增加诈唬成功权重
    reward_calculator.reward_weights['hand_strength'] = 0.4    # 增加手牌强度权重
    reward_calculator.reward_weights['fold_when_weak'] = 0.5   # 增加及时弃牌奖励
    
    return game, players, ai_agents, reward_calculator

def run_test_hands(game, players, ai_agents, reward_calculator, num_hands=1000):
    """运行指定数量的测试手牌"""
    logger.info(f"开始测试，共运行 {num_hands} 手牌")
    
    # 初始化统计数据
    hand_results = []
    profit_history = {player.name: [] for player in players}
    
    # 记录初始筹码
    initial_chips = {player.name: player.chips for player in players}
    
    # 运行测试手牌
    for hand_num in range(1, num_hands + 1):
        # 进度显示
        if hand_num % 100 == 0:
            logger.info(f"已完成 {hand_num}/{num_hands} 手牌")
        
        # 开始新手牌
        game.start_new_hand()
        
        # 记录每轮开始时的玩家筹码
        start_chips = {player.name: player.chips for player in players}
        
        # 进行游戏
        while not game.is_hand_over:
            # 获取当前活跃玩家
            current_player = game.get_current_player()
            
            if current_player.is_ai and current_player.name in ai_agents:
                # AI玩家选择动作
                action, amount = ai_agents[current_player.name].select_action(game, current_player)
            else:
                # 非AI玩家采取默认动作（简单策略）
                if game.current_bet > current_player.bet:
                    # 如果有下注，有30%概率跟注，70%概率弃牌
                    if np.random.random() < 0.3:
                        action = 'call'
                        amount = game.current_bet - current_player.bet
                    else:
                        action = 'fold'
                        amount = None
                else:
                    # 如果没有下注，有50%概率check，50%概率小额加注
                    if np.random.random() < 0.5:
                        action = 'check'
                        amount = None
                    else:
                        action = 'raise'
                        amount = game.current_bet + game.big_blind * 2
            
            # 执行动作
            game.execute_action(current_player, action, amount)
        
        # 计算每轮利润并更新累计利润
        round_profits = {}
        for player in players:
            round_profit = player.chips - start_chips[player.name]
            round_profits[player.name] = round_profit
            player.total_profit += round_profit
            profit_history[player.name].append(player.total_profit)
        
        # 记录结果
        hand_results.append({
            'hand_num': hand_num,
            'winner': game.hand_winner.name if game.hand_winner else 'None',
            'profits': round_profits
        })
    
    # 计算最终利润
    final_profits = {player.name: player.chips - initial_chips[player.name] 
                    for player in players}
    
    return hand_results, profit_history, final_profits

def analyze_results(players, hand_results, final_profits):
    """分析测试结果"""
    logger.info("\n===== 测试结果分析 =====")
    
    # 打印最终利润
    logger.info("\n最终利润统计：")
    for player_name, profit in sorted(final_profits.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"{player_name}: {profit:+}")
    
    # 分析AI玩家表现
    ai_players = [p for p in players if p.is_ai]
    ai_profits = [p.total_profit for p in ai_players]
    
    avg_ai_profit = np.mean(ai_profits)
    std_ai_profit = np.std(ai_profits)
    
    logger.info(f"\nAI玩家平均利润: {avg_ai_profit:+}")
    logger.info(f"AI玩家利润标准差: {std_ai_profit}")
    
    # 计算胜率
    winner_counts = {player.name: 0 for player in players}
    for result in hand_results:
        if result['winner'] in winner_counts:
            winner_counts[result['winner']] += 1
    
    logger.info("\n玩家胜率：")
    for player_name, wins in sorted(winner_counts.items(), key=lambda x: x[1], reverse=True):
        win_rate = wins / len(hand_results) * 100
        logger.info(f"{player_name}: {wins}胜 ({win_rate:.2f}%)")
    
    # 评估修复是否成功
    if avg_ai_profit > 0:
        logger.info("\n✅ 修复成功：AI玩家平均利润为正")
    else:
        logger.info("\n❌ 修复未成功：AI玩家平均利润仍为负")
        
    # 分析最盈利和最亏损的玩家
    best_player = max(players, key=lambda p: p.total_profit)
    worst_player = min(players, key=lambda p: p.total_profit)
    
    logger.info(f"\n表现最佳玩家: {best_player.name} (利润: {best_player.total_profit:+})")
    logger.info(f"表现最差玩家: {worst_player.name} (利润: {worst_player.total_profit:+})")

def main():
    """主函数"""
    try:
        start_time = time.time()
        
        # 设置测试环境
        logger.info("设置测试环境...")
        game, players, ai_agents, reward_calculator = setup_test_environment()
        
        # 运行测试
        logger.info("开始运行测试...")
        hand_results, profit_history, final_profits = run_test_hands(
            game, players, ai_agents, reward_calculator, num_hands=1000
        )
        
        # 分析结果
        analyze_results(players, hand_results, final_profits)
        
        # 保存结果到文件
        np.save('profit_history.npy', profit_history)
        logger.info("\n利润历史已保存到 profit_history.npy")
        
        end_time = time.time()
        logger.info(f"\n测试完成！总用时: {end_time - start_time:.2f} 秒")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    main()