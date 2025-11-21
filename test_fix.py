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
logging.basicConfig(level=logging.DEBUG,  # 修改为DEBUG级别以查看更详细的日志
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("test_fix.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger('TestFix')
logger.setLevel(logging.DEBUG)  # 确保logger使用DEBUG级别

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
    
    # 创建AI代理（使用PPO模型）
    ai_agents = {}
    for player in players:
        if player.is_ai:
            ai_agents[player.name] = PokerAIAgent(agent_type='ppo')
    
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
    logger.debug("初始化统计数据完成")
    
    # 记录初始筹码
    initial_chips = {player.name: player.stack for player in players}
    
    # 运行测试手牌
    for hand_num in range(1, num_hands + 1):
        # 进度显示
        if hand_num % 100 == 0:
            logger.info(f"已完成 {hand_num}/{num_hands} 手牌")
        elif hand_num % 10 == 0:
            logger.debug(f"处理手牌 {hand_num}/{num_hands}")
        
        # 开始新手牌
        logger.debug(f"开始手牌 #{hand_num}")
        game.start_new_hand()
        
        # 记录每轮开始时的玩家筹码
        start_chips = {player.name: player.stack for player in players}
        
        # 进行游戏
        logger.debug(f"开始游戏循环，初始玩家索引: {game.current_player}")
        loop_count = 0
        max_loop_iterations = 100  # 防止无限循环
        while not game.is_hand_over():
            loop_count += 1
            if loop_count > max_loop_iterations:
                logger.error(f"警告: 游戏循环超过{max_loop_iterations}次迭代，可能陷入死循环")
                logger.error(f"当前状态: hand_num={hand_num}, current_player_index={game.current_player}, current_bet={game.current_bet}")
                # 显示所有玩家状态
                for i, p in enumerate(game.players):
                    logger.error(f"玩家{i} ({p.name}): stack={p.stack}, bet={p.bet}, is_active={p.is_active}, is_all_in={p.is_all_in}")
                break
            
            # 检查当前轮次是否结束，如果结束则进入下一轮
            if game.is_round_over():
                logger.debug(f"轮次{game.round}结束，进入下一轮")
                is_hand_over = game.proceed_to_next_round()
                if is_hand_over:
                    logger.debug("手牌结束")
                    break
                continue
                
            # 获取当前活跃玩家
            current_player = game.players[game.current_player]
            # 跳过非活跃或已全下的玩家
            if not current_player.is_active or current_player.is_all_in:
                logger.debug(f"跳过非活跃或已全下的玩家: {current_player.name}")
                game.next_player()
                continue
                
            logger.debug(f"循环#{loop_count}: 当前玩家={current_player.name}, 索引={game.current_player}, 已下注={current_player.bet}, 当前底池={game.current_bet}")
            
            # 确定安全的最大下注金额，避免过大的下注
            max_safe_bet = min(current_player.stack, game.big_blind * 5)  # 限制最大下注为大盲注的5倍
            
            if current_player.is_ai and current_player.name in ai_agents:
                # AI玩家选择动作
                logger.debug(f"AI玩家 {current_player.name} 正在选择动作")
                action, amount = ai_agents[current_player.name].select_action(game, current_player)
                # 限制AI的下注金额
                if action == 'raise' and amount is not None and amount > max_safe_bet:
                    amount = max_safe_bet
                    logger.debug(f"限制AI下注金额为: {amount}")
                logger.debug(f"AI玩家 {current_player.name} 选择动作: {action}, 金额: {amount}")
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
                        # 使用安全的小额加注
                        amount = min(game.current_bet + game.big_blind * 2, max_safe_bet)
                logger.debug(f"非AI玩家 {current_player.name} 选择动作: {action}, 金额: {amount}")
            
            # 执行动作
            logger.debug(f"执行动作: {action}, 金额: {amount}")
            game.handle_action(current_player, action, amount)
            logger.debug(f"动作执行完成")
            
            # 移动到下一个玩家
            logger.debug(f"移动到下一个玩家")
            next_player = game.next_player()
            if next_player is None:
                logger.debug("没有找到下一个活跃玩家，检查轮次是否结束")
                if game.is_round_over():
                    game.proceed_to_next_round()
            logger.debug(f"新的当前玩家索引: {game.current_player}")
        
        if loop_count > max_loop_iterations:
            logger.error("由于可能的死循环，提前结束手牌")
            # 重置游戏状态以继续下一手牌
            game.start_new_hand()
        
        # 计算每轮利润并更新累计利润
        logger.debug("计算本轮利润")
        round_profits = {}
        for player in players:
            round_profit = player.stack - start_chips[player.name]
            round_profits[player.name] = round_profit
            player.total_profit += round_profit
            profit_history[player.name].append(player.total_profit)
            logger.debug(f"玩家 {player.name}: 本轮利润={round_profit}, 累计利润={player.total_profit}")
        
        # 记录结果（由于PokerGame类没有hand_winner属性，暂时记录为Unknown）
        # 查找当前轮次中筹码增加的玩家作为获胜者
        logger.debug("确定本轮获胜者")
        winner = None
        for player in players:
            if player.stack > start_chips[player.name]:
                winner = player
                break
        logger.debug(f"手牌 #{hand_num} 获胜者: {winner.name if winner else 'Unknown'}")
        
        hand_results.append({
            'hand_num': hand_num,
            'winner': winner.name if winner else 'Unknown',
            'profits': round_profits
        })
    
    # 计算最终利润
    logger.debug("计算最终利润")
    final_profits = {player.name: player.stack - initial_chips[player.name] 
                    for player in players}
    logger.debug(f"最终利润: {final_profits}")
    
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