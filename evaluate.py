#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
德州扑克AI模型详细评估脚本

此脚本用于对训练好的德州扑克AI模型进行全面评估，
生成详细的性能报告和可视化图表。
"""

import os
import argparse
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("eval_detail_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('PokerAIEvaluator')

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    logger.info("已设置中文字体支持")
except Exception as e:
    logger.warning(f"设置中文字体支持失败: {e}")

# 导入必要的模块
try:
    from poker_ai.reward.self_play_trainer import SelfPlayTrainer
    from poker_ai.advisor.ai_advisor import AIAvisor
    from poker_ai.game.poker_game import PokerGame, Player
    from poker_ai.model.ai_agent import PokerAIAgent
    from poker_ai.model.hand_evaluator import HandEvaluator
    logger.info("成功导入所有必要模块")
except Exception as e:
    logger.error(f"导入模块失败: {e}")
    raise

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='德州扑克AI模型详细评估脚本')
    
    # 模型参数
    parser.add_argument('--model-path', type=str, required=True,
                        help='要评估的模型路径')
    parser.add_argument('--model-type', type=str, choices=['dqn', 'policy_gradient'],
                        default='dqn', help='模型类型')
    
    # 评估参数
    parser.add_argument('--num-games', type=int, default=100,
                        help='评估的游戏局数')
    parser.add_argument('--num-opponents', type=int, default=5,
                        help='对手数量')
    parser.add_argument('--opponent-type', type=str, choices=['random', 'fixed', 'previous'],
                        default='fixed', help='对手类型')
    
    # 游戏参数
    parser.add_argument('--small-blind', type=int, default=5, help='小盲注')
    parser.add_argument('--big-blind', type=int, default=10, help='大盲注')
    parser.add_argument('--initial-stack', type=int, default=1000, help='初始筹码量')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='./eval_reports',
                        help='评估报告输出目录')
    parser.add_argument('--generate-plots', action='store_true',
                        help='生成可视化图表')
    parser.add_argument('--verbose', action='store_true', help='输出详细日志')
    
    return parser.parse_args()

def setup_environment(args):
    """
    设置评估环境
    """
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 如果启用详细日志，更新日志级别
    if args.verbose:
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    logger.info(f"环境设置完成: 输出目录={args.output_dir}")

def create_opponents(args, num_opponents):
    """
    创建对手代理
    """
    opponents = []
    
    for i in range(num_opponents):
        if args.opponent_type == 'random':
            # 创建随机对手代理
            opponent = PokerAIAgent(agent_type='random')
        elif args.opponent_type == 'previous':
            # 创建使用历史模型的对手
            opponent = PokerAIAgent(agent_type=args.model_type)
            # 这里简化处理，实际应该加载历史版本的模型
        else:  # fixed
            # 创建固定策略对手
            opponent = PokerAIAgent(agent_type='fixed')
        
        opponents.append(opponent)
    
    return opponents

def run_comprehensive_evaluation(args):
    """
    运行全面评估
    """
    logger.info("开始全面评估德州扑克AI模型")
    logger.info(f"评估参数: {vars(args)}")
    
    # 创建训练器用于评估
    trainer = SelfPlayTrainer(
        model_type=args.model_type,
        checkpoint_dir=args.output_dir,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        initial_stack=args.initial_stack,
        num_players=args.num_opponents + 1,  # +1 for the main agent
        num_eval_games=args.num_games
    )
    
    # 加载模型
    success = trainer.load_checkpoint(args.model_path)
    if not success:
        logger.error(f"无法加载模型: {args.model_path}")
        return None
    
    # 执行基本评估
    logger.info(f"执行基本评估，共进行 {args.num_games} 局游戏...")
    basic_results = trainer.evaluate()
    
    # 运行详细评估
    detailed_results = run_detailed_evaluation(args, trainer)
    
    # 合并结果
    evaluation_results = {
        'basic': basic_results,
        'detailed': detailed_results,
        'metadata': {
            'model_path': args.model_path,
            'model_type': args.model_type,
            'evaluation_time': datetime.now().isoformat(),
            'num_games': args.num_games,
            'game_parameters': {
                'small_blind': args.small_blind,
                'big_blind': args.big_blind,
                'initial_stack': args.initial_stack,
                'num_players': args.num_opponents + 1
            }
        }
    }
    
    return evaluation_results

def run_detailed_evaluation(args, trainer):
    """
    运行详细评估，收集更多统计信息
    """
    logger.info("运行详细评估...")
    
    # 初始化统计数据
    detailed_stats = {
        'game_results': [],
        'hand_strength_distribution': [],
        'action_distribution': {'fold': 0, 'call': 0, 'raise': 0},
        'profit_by_stage': {'preflop': [], 'flop': [], 'turn': [], 'river': []},
        'win_by_position': {i: 0 for i in range(args.num_opponents + 1)},
        'win_rate_by_hand_strength': {
            'weak': {'count': 0, 'wins': 0},
            'medium': {'count': 0, 'wins': 0},
            'strong': {'count': 0, 'wins': 0}
        }
    }
    
    # 运行指定数量的游戏
    for game_idx in range(args.num_games):
        logger.debug(f"评估游戏 {game_idx + 1}/{args.num_games}")
        
        # 创建游戏
        game, players = trainer._create_game()
        
        # 记录游戏结果
        game_result = {
            'game_id': game_idx + 1,
            'actions': [],
            'stage_results': {},
            'final_stack': players[0].stack,  # 主玩家的最终筹码
            'is_winner': False
        }
        
        # 开始新的一手牌
        game.start_new_hand()
        
        # 记录初始手牌强度（翻牌前）
        main_player = players[0]
        hand_evaluator = HandEvaluator()
        preflop_strength, _ = hand_evaluator.evaluate_hand(main_player.cards)
        detailed_stats['hand_strength_distribution'].append(preflop_strength)
        
        # 分类手牌强度
        strength_category = 'weak'
        if preflop_strength > 0.7:
            strength_category = 'strong'
        elif preflop_strength > 0.4:
            strength_category = 'medium'
        
        detailed_stats['win_rate_by_hand_strength'][strength_category]['count'] += 1
        
        # 游戏主循环
        while True:
            # 记录当前轮次
            current_stage = ['preflop', 'flop', 'turn', 'river'][game.round_num]
            stage_start_stack = main_player.stack
            
            # 处理当前轮次的所有动作
            while not game.is_round_over():
                current_player = game.next_player()
                if not current_player:
                    break
                
                # 决定使用哪个代理
                if current_player.name == "Main_AI":
                    agent = trainer.main_agent
                else:
                    # 为对手选择一个代理
                    opponent_idx = int(current_player.name.split('_')[-1]) - 1
                    if opponent_idx < len(trainer.opponent_agents):
                        agent = trainer.opponent_agents[opponent_idx]
                    else:
                        agent = trainer.opponent_agents[0]
                
                # 选择动作
                action, amount = agent.select_action(game, current_player)
                
                # 记录主玩家的动作
                if current_player.name == "Main_AI":
                    game_result['actions'].append({
                        'stage': current_stage,
                        'action': action,
                        'amount': amount,
                        'pot_size': game.pot
                    })
                    
                    # 更新动作分布
                    if action in detailed_stats['action_distribution']:
                        detailed_stats['action_distribution'][action] += 1
            
            # 记录轮次结果
            stage_profit = main_player.stack - stage_start_stack
            game_result['stage_results'][current_stage] = stage_profit
            detailed_stats['profit_by_stage'][current_stage].append(stage_profit)
            
            # 进入下一轮或结束游戏
            if game.proceed_to_next_round():
                break
        
        # 记录游戏结果
        if hasattr(game, 'winner') and game.winner and game.winner.name == "Main_AI":
            game_result['is_winner'] = True
            detailed_stats['win_by_position'][0] += 1  # 位置0是主玩家
            detailed_stats['win_rate_by_hand_strength'][strength_category]['wins'] += 1
        
        detailed_stats['game_results'].append(game_result)
    
    # 计算胜率按手牌强度
    for category in detailed_stats['win_rate_by_hand_strength']:
        stats = detailed_stats['win_rate_by_hand_strength'][category]
        if stats['count'] > 0:
            stats['win_rate'] = stats['wins'] / stats['count']
        else:
            stats['win_rate'] = 0.0
    
    return detailed_stats

def save_evaluation_results(args, results):
    """
    保存评估结果到文件
    """
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(args.output_dir, f"eval_results_{timestamp}.json")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 将numpy类型转换为Python原生类型以便序列化
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    # 转换数据
    serializable_results = convert_numpy(results)
    
    # 保存到文件
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        logger.info(f"评估结果已保存到: {filename}")
        return filename
    except Exception as e:
        logger.error(f"保存评估结果失败: {e}")
        return None

def generate_evaluation_report(args, results, results_file):
    """
    生成评估报告
    """
    # 创建报告内容
    report = f"""
    ======================================================
                    德州扑克AI模型详细评估报告
    ======================================================
    评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    模型路径: {args.model_path}
    模型类型: {args.model_type}
    评估游戏数: {args.num_games}
    
    基本评估结果:
    - 平均利润: {results['basic']['avg_profit']:.2f}
    - 胜率: {results['basic']['win_rate']:.2%}
    - 总利润: {results['basic']['total_profit']:.2f}
    - 获胜次数: {results['basic']['wins']}
    
    详细统计:
    1. 动作分布:
       - 弃牌: {results['detailed']['action_distribution']['fold']}次
       - 跟注: {results['detailed']['action_distribution']['call']}次
       - 加注: {results['detailed']['action_distribution']['raise']}次
    
    2. 各阶段平均利润:
       - 翻牌前: {np.mean(results['detailed']['profit_by_stage']['preflop']):.2f}
       - 翻牌: {np.mean(results['detailed']['profit_by_stage']['flop']):.2f}
       - 转牌: {np.mean(results['detailed']['profit_by_stage']['turn']):.2f}
       - 河牌: {np.mean(results['detailed']['profit_by_stage']['river']):.2f}
    
    3. 按手牌强度的胜率:
       - 弱牌 (<= 0.4): {results['detailed']['win_rate_by_hand_strength']['weak']['win_rate']:.2%} 
         (样本数: {results['detailed']['win_rate_by_hand_strength']['weak']['count']})
       - 中等牌 (0.4-0.7): {results['detailed']['win_rate_by_hand_strength']['medium']['win_rate']:.2%} 
         (样本数: {results['detailed']['win_rate_by_hand_strength']['medium']['count']})
       - 强牌 (> 0.7): {results['detailed']['win_rate_by_hand_strength']['strong']['win_rate']:.2%} 
         (样本数: {results['detailed']['win_rate_by_hand_strength']['strong']['count']})
    
    4. 按位置的胜率:
       - 主玩家: {results['detailed']['win_by_position'][0] / args.num_games:.2%}
    
    评估结果文件: {results_file}
    ======================================================
    """
    
    # 打印报告
    print(report)
    
    # 保存报告到文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(args.output_dir, f"eval_report_{timestamp}.txt")
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"评估报告已保存到: {report_file}")
        return report_file
    except Exception as e:
        logger.error(f"保存评估报告失败: {e}")
        return None

def generate_visualizations(args, results):
    """
    生成可视化图表
    """
    if not args.generate_plots:
        logger.info("未启用图表生成")
        return []
    
    logger.info("生成可视化图表...")
    
    # 确保输出目录存在
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    saved_plots = []
    
    # 1. 手牌强度分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(results['detailed']['hand_strength_distribution'], bins=20, alpha=0.7, color='skyblue')
    plt.title('手牌强度分布')
    plt.xlabel('手牌强度')
    plt.ylabel('频率')
    plt.grid(True, alpha=0.3)
    
    plot_file = os.path.join(plots_dir, 'hand_strength_distribution.png')
    plt.savefig(plot_file)
    plt.close()
    saved_plots.append(plot_file)
    logger.info(f"已保存手牌强度分布图: {plot_file}")
    
    # 2. 动作分布饼图
    actions = results['detailed']['action_distribution']
    labels = list(actions.keys())
    sizes = list(actions.values())
    colors = ['lightcoral', 'lightskyblue', 'lightgreen']
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # 确保饼图是圆的
    plt.title('动作分布')
    
    plot_file = os.path.join(plots_dir, 'action_distribution.png')
    plt.savefig(plot_file)
    plt.close()
    saved_plots.append(plot_file)
    logger.info(f"已保存动作分布图: {plot_file}")
    
    # 3. 各阶段利润箱线图
    profit_by_stage = results['detailed']['profit_by_stage']
    stages = list(profit_by_stage.keys())
    data = [profit_by_stage[stage] for stage in stages]
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data)
    plt.title('各阶段利润分布')
    plt.xticks(range(len(stages)), ['翻牌前', '翻牌', '转牌', '河牌'])
    plt.ylabel('利润')
    plt.grid(True, alpha=0.3)
    
    plot_file = os.path.join(plots_dir, 'profit_by_stage.png')
    plt.savefig(plot_file)
    plt.close()
    saved_plots.append(plot_file)
    logger.info(f"已保存各阶段利润分布图: {plot_file}")
    
    # 4. 按手牌强度的胜率条形图
    strength_win_rates = results['detailed']['win_rate_by_hand_strength']
    categories = ['弱牌', '中等牌', '强牌']
    win_rates = [
        strength_win_rates['weak']['win_rate'],
        strength_win_rates['medium']['win_rate'],
        strength_win_rates['strong']['win_rate']
    ]
    counts = [
        strength_win_rates['weak']['count'],
        strength_win_rates['medium']['count'],
        strength_win_rates['strong']['count']
    ]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, win_rates, color=['red', 'orange', 'green'])
    
    # 在柱状图上添加样本数量
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'n={counts[i]}', ha='center', va='bottom')
    
    plt.title('按手牌强度的胜率')
    plt.ylabel('胜率')
    plt.ylim(0, 1.1)  # 设置y轴范围以便更好地查看
    plt.grid(True, axis='y', alpha=0.3)
    
    plot_file = os.path.join(plots_dir, 'win_rate_by_strength.png')
    plt.savefig(plot_file)
    plt.close()
    saved_plots.append(plot_file)
    logger.info(f"已保存按手牌强度的胜率图: {plot_file}")
    
    # 5. 累计利润曲线
    game_profits = [game['final_stack'] - args.initial_stack for game in results['detailed']['game_results']]
    cumulative_profits = np.cumsum(game_profits)
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, args.num_games + 1), cumulative_profits, 'b-')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('累计利润曲线')
    plt.xlabel('游戏局数')
    plt.ylabel('累计利润')
    plt.grid(True, alpha=0.3)
    
    plot_file = os.path.join(plots_dir, 'cumulative_profit.png')
    plt.savefig(plot_file)
    plt.close()
    saved_plots.append(plot_file)
    logger.info(f"已保存累计利润曲线图: {plot_file}")
    
    return saved_plots

def main():
    """
    主函数
    """
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 设置环境
        setup_environment(args)
        
        # 运行评估
        results = run_comprehensive_evaluation(args)
        if results is None:
            logger.error("评估失败")
            return 1
        
        # 保存评估结果
        results_file = save_evaluation_results(args, results)
        if results_file is None:
            logger.error("保存评估结果失败")
            return 1
        
        # 生成评估报告
        report_file = generate_evaluation_report(args, results, results_file)
        
        # 生成可视化图表
        if args.generate_plots:
            try:
                saved_plots = generate_visualizations(args, results)
                logger.info(f"已生成 {len(saved_plots)} 个可视化图表")
            except Exception as e:
                logger.error(f"生成可视化图表失败: {e}")
        
        logger.info("评估完成！")
        return 0
    except KeyboardInterrupt:
        logger.info("用户中断了程序")
        return 1
    except Exception as e:
        logger.error(f"程序执行过程中发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)