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
import platform
import numpy as np
from datetime import datetime

# ----- 日志（先建好，后续所有 import 出现的告警都能落盘）-----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("eval_detail_log.txt", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('PokerAIEvaluator')

# ----- TensorFlow 初始化：必须在 import poker_ai.* 之前设置 GPU 显存增长 -----
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logger.info(f"已启用 GPU 显存按需增长 (检测到 {len(gpus)} 块 GPU，仅使用第 0 块)")
        except RuntimeError as e:
            logger.warning(f"设置 GPU 显存增长失败 (TF 已初始化?): {e}")
    else:
        logger.info("未检测到可用 GPU，将使用 CPU")
except ImportError:
    logger.warning("未安装 TensorFlow，部分功能不可用")
    tf = None  # 让后续代码可感知

# ----- matplotlib 中文字体：跨平台兜底 -----
import matplotlib
matplotlib.use("Agg")  # 评估脚本无需 GUI，避免 Windows/远程环境弹窗或报错
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns


def _setup_chinese_font() -> None:
    """跨平台设置中文字体，找不到时静默降级，不再警告刷屏。"""
    candidates_by_os = {
        "Windows": ["Microsoft YaHei", "SimHei", "DengXian"],
        "Darwin":  ["PingFang SC", "Heiti SC", "Arial Unicode MS"],
        "Linux":   ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "Source Han Sans SC"],
    }
    candidates = candidates_by_os.get(platform.system(), [])
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), None)
    if chosen:
        plt.rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        logger.info(f"matplotlib 中文字体已设置为: {chosen}")
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        logger.info("未找到中文字体，图表将使用 DejaVu Sans (中文可能显示为方块)")


_setup_chinese_font()

# ----- 业务模块导入 -----
try:
    from poker_ai.reward.self_play_trainer import SelfPlayTrainer
    from poker_ai.advisor.ai_advisor import AIAvisor
    from poker_ai.game.poker_game import PokerGame, Player
    from poker_ai.game.cards import HAND_RANKS as CARDS_HAND_RANKS
    from poker_ai.model.ai_agent import PokerAIAgent
    from poker_ai.model.hand_evaluator import HandEvaluator
    logger.info("成功导入所有必要模块")
except ImportError as e:
    if "tensorflow" in str(e).lower():
        logger.error(f"缺少 TensorFlow 依赖: {e}")
        logger.error("请先安装 TensorFlow: pip install tensorflow")
    else:
        logger.error(f"导入模块失败: {e}")
    raise

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='德州扑克AI模型详细评估脚本')
    
    # 模型参数
    parser.add_argument('--model-path', type=str, required=True,
                        help='要评估的模型路径（PPO 请传 base 路径，不带 .actor.weights.h5 后缀）')
    parser.add_argument('--model-type', type=str, choices=['dqn', 'policy_gradient', 'ppo'],
                        default='dqn', help='模型类型')

    # 评估参数
    parser.add_argument('--num-games', type=int, default=100,
                        help='评估的游戏局数')
    parser.add_argument('--num-opponents', type=int, default=5,
                        help='对手数量（注意：本项目固定 6-Max，因此通常应为 5）')
    parser.add_argument('--opponent-type', type=str, choices=['previous'],
                        default='previous',
                        help='对手类型（当前仅支持使用同模型类型的副本作为对手）')
    
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
    创建对手代理。

    注意：当前 PokerAIAgent 仅支持 'dqn' / 'policy_gradient' / 'ppo' 三类，
    早期版本提到的 'random' / 'fixed' 类型已废弃，统一使用同模型类型的副本作为对手。
    """
    opponents = []
    for _ in range(num_opponents):
        opponents.append(PokerAIAgent(agent_type=args.model_type))
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
    
    # 加载模型 - 改进路径处理以确保能正确加载checkpoints目录中的模型文件
    try:
        # 确保路径是绝对路径
        model_path = os.path.abspath(args.model_path)
        logger.info(f"尝试加载模型: {model_path}")
        
        # 详细检查文件信息
        logger.info(f"文件存在性检查: {os.path.exists(model_path)}")
        if os.path.exists(model_path):
            logger.info(f"文件大小: {os.path.getsize(model_path)} 字节")
            logger.info(f"文件权限: {os.access(model_path, os.R_OK)}")
        
        # 检查路径是否存在
        if not os.path.exists(model_path):
            # 尝试在checkpoints目录中查找
            checkpoint_path = os.path.join('checkpoints', args.model_path)
            if os.path.exists(checkpoint_path):
                model_path = os.path.abspath(checkpoint_path)
                logger.info(f"在checkpoints目录中找到模型: {model_path}")
            else:
                logger.error(f"模型文件不存在: {args.model_path}")
                logger.info(f"当前工作目录: {os.getcwd()}")
                if os.path.exists('checkpoints'):
                    logger.info(f"checkpoints目录内容: {os.listdir('checkpoints')}")
                return None
        
        # 直接使用trainer.main_agent.load_model方法（无返回值）
        logger.info(f"调用trainer.main_agent.load_model方法")
        trainer.main_agent.load_model(model_path)
        logger.info(f"模型加载完成: {model_path}")
        # 移除未使用的success变量
    except Exception as e:
        logger.error(f"加载模型时发生错误: {str(e)}")
        logger.error(f"错误类型: {type(e).__name__}")
        import traceback
        logger.error(f"错误堆栈: {traceback.format_exc()}")
        logger.info(f"当前工作目录: {os.getcwd()}")
        if os.path.exists('checkpoints'):
            logger.info(f"checkpoints目录内容: {os.listdir('checkpoints')}")
        if os.path.exists(model_path):
            logger.info(f"模型文件确实存在，权限: {os.access(model_path, os.R_OK)}")
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
        # 每10局游戏打印一次进度，确保程序仍在运行
        if game_idx % 10 == 0:
            logger.info(f"评估进度: 已完成 {game_idx}/{args.num_games} 局游戏")
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
        logger.info("手牌评估器已初始化")  # 添加明确的日志标记
        
        # 翻牌前阶段没有公共牌，传入空列表作为community_cards
        hand_result, _ = hand_evaluator.evaluate_hand(main_player.cards, [])
        # 将牌型转换为数值强度（HAND_RANKS 在 poker_ai.game.cards 模块级，不在 HandEvaluator 类上）
        preflop_strength = CARDS_HAND_RANKS.get(hand_result, 0) / 9.0  # 归一化到 0-1 范围
        detailed_stats['hand_strength_distribution'].append(preflop_strength)
        
        # 分类手牌强度
        strength_category = 'weak'
        if preflop_strength > 0.7:
            strength_category = 'strong'
        elif preflop_strength > 0.4:
            strength_category = 'medium'
        
        detailed_stats['win_rate_by_hand_strength'][strength_category]['count'] += 1
        
        # 游戏主循环
        max_rounds = 10  # 设置最大轮数限制，防止无限循环
        round_count = 0
        
        while round_count < max_rounds:
            round_count += 1
            
            # 记录当前轮次 - 添加安全检查
            try:
                if 1 <= game.round <= 4:
                    current_stage = ['preflop', 'flop', 'turn', 'river'][game.round - 1]
                    logger.debug(f"游戏{game_idx + 1} - 当前阶段: {current_stage}")
                else:
                    current_stage = 'unknown'
                    logger.warning(f"游戏{game_idx + 1} - 无效的游戏轮次: {game.round}")
            except Exception as e:
                current_stage = 'unknown'
                logger.error(f"游戏{game_idx + 1} - 获取游戏轮次时出错: {e}")
            
            stage_start_stack = main_player.stack
            
            # 处理当前轮次的所有动作，添加最大动作限制
            max_actions = 100  # 每轮最大动作数限制
            action_count = 0
            
            try:
                while not game.is_round_over() and action_count < max_actions:
                    action_count += 1
                    
                    current_player = game.next_player()
                    if not current_player:
                        logger.warning(f"游戏{game_idx + 1}在{current_stage}阶段没有更多玩家")
                        break
                    
                    # 决定使用哪个代理
                    if current_player.name == "Main_AI":
                        agent = trainer.main_agent
                    else:
                        # 为对手选择一个代理
                        try:
                            # 安全地获取对手索引
                            name_parts = current_player.name.split('_')
                            if len(name_parts) > 1 and name_parts[-1].isdigit():
                                opponent_idx = int(name_parts[-1]) - 1
                                # 确保索引有效
                                if 0 <= opponent_idx < len(trainer.opponent_agents):
                                    agent = trainer.opponent_agents[opponent_idx]
                                else:
                                    agent = trainer.opponent_agents[0]  # 默认使用第一个对手代理
                            else:
                                agent = trainer.opponent_agents[0]  # 默认使用第一个对手代理
                        except Exception as e:
                            logger.warning(f"游戏{game_idx + 1} - 选择对手代理时出错: {e}，使用默认代理")
                            agent = trainer.opponent_agents[0]
                    
                    # 选择动作
                    try:
                        action, amount = agent.select_action(game, current_player)
                    except Exception as e:
                        logger.error(f"游戏{game_idx + 1} - 代理选择动作时出错: {e}")
                        action, amount = 'fold', 0  # 使用默认动作避免崩溃
                    
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
                
                if action_count >= max_actions:
                    logger.warning(f"游戏{game_idx + 1}在{current_stage}阶段达到最大动作限制")
            except Exception as e:
                logger.error(f"处理游戏{game_idx + 1}的{current_stage}阶段时出错: {e}")
            
            # 记录轮次结果 - 添加详细日志
            try:
                stage_profit = main_player.stack - stage_start_stack
                logger.debug(f"游戏{game_idx + 1} - {current_stage}阶段利润: {stage_profit}")
                
                # 确保stage_results字典存在
                if 'stage_results' not in game_result:
                    game_result['stage_results'] = {}
                game_result['stage_results'][current_stage] = stage_profit
                
                # 安全地添加到详细统计中
                if current_stage in detailed_stats['profit_by_stage']:
                    detailed_stats['profit_by_stage'][current_stage].append(stage_profit)
                    logger.debug(f"游戏{game_idx + 1} - 已将{current_stage}阶段利润添加到统计中，当前累计: {detailed_stats['profit_by_stage'][current_stage]}")
                else:
                    logger.warning(f"游戏{game_idx + 1} - 未知的游戏阶段: {current_stage}")
            except Exception as e:
                logger.error(f"游戏{game_idx + 1} - 记录轮次结果时出错: {e}")
            
            # 进入下一轮或结束游戏
            try:
                # 添加安全检查：确保对手数量大于0
                if len(game.players) <= 1:
                    logger.info(f"游戏{game_idx + 1}因玩家数量不足而结束")
                    break
                
                # 正确处理游戏进行和结束的逻辑
                # 修改这里的逻辑：当轮次完成后，继续下一轮或结束游戏
                # 如果game.proceed_to_next_round()返回True表示游戏结束
                if game.proceed_to_next_round():
                    # 游戏结束，跳出循环
                    logger.debug(f"游戏{game_idx + 1}已结束")
                    break
            except Exception as e:
                logger.error(f"游戏{game_idx + 1}进行下一轮时出错: {e}")
                # 发生错误时也跳出循环以防止无限循环
                break
        
        if round_count >= max_rounds:
            logger.warning(f"游戏{game_idx + 1}达到最大轮数限制，强制结束")
        
        # 记录游戏结果
        try:
            winner_names = [p.name for p in getattr(game, 'winners', [])]
            if not winner_names and hasattr(game, 'winner') and game.winner:
                winner_names = [game.winner.name]

            if "Main_AI" in winner_names:
                game_result['is_winner'] = True
                detailed_stats['win_by_position'][0] += 1  # 位置0是主玩家
                detailed_stats['win_rate_by_hand_strength'][strength_category]['wins'] += 1
        except Exception as e:
            logger.error(f"游戏{game_idx + 1}记录获胜结果时出错: {e}")
        
        detailed_stats['game_results'].append(game_result)
    
    # 计算胜率按手牌强度
    for category in detailed_stats['win_rate_by_hand_strength']:
        stats = detailed_stats['win_rate_by_hand_strength'][category]
        if stats['count'] > 0:
            stats['win_rate'] = stats['wins'] / stats['count']
        else:
            stats['win_rate'] = 0.0
    
    # 确保各阶段利润统计正确，处理空列表情况
    for stage in detailed_stats['profit_by_stage']:
        if not detailed_stats['profit_by_stage'][stage]:
            # 如果阶段利润列表为空，添加一个0作为默认值
            detailed_stats['profit_by_stage'][stage] = [0.0]
            logger.warning(f"阶段 {stage} 没有利润数据，设置默认值为0")
    
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
    
   # 基本评估结果:
    - 平均利润: {results.get('basic', {}).get('avg_profit', 0):.2f}
    - 胜率: {results.get('basic', {}).get('win_rate', 0):.2%}
    - 总利润: {results.get('basic', {}).get('total_profit', 0):.2f}
    - 获胜次数: {results.get('basic', {}).get('wins', 0)}
    
    详细统计:
    1. 动作分布:
       - 弃牌: {results.get('detailed', {}).get('action_distribution', {}).get('fold', 0)}次
       - 跟注: {results.get('detailed', {}).get('action_distribution', {}).get('call', 0)}次
       - 加注: {results.get('detailed', {}).get('action_distribution', {}).get('raise', 0)}次
    
    2. 各阶段平均利润:
       - 翻牌前: {np.mean(results.get('detailed', {}).get('profit_by_stage', {}).get('preflop', [0])):.2f}
       - 翻牌: {np.mean(results.get('detailed', {}).get('profit_by_stage', {}).get('flop', [0])):.2f}
       - 转牌: {np.mean(results.get('detailed', {}).get('profit_by_stage', {}).get('turn', [0])):.2f}
       - 河牌: {np.mean(results.get('detailed', {}).get('profit_by_stage', {}).get('river', [0])):.2f}
    
    3. 按手牌强度的胜率:
       - 弱牌 (<= 0.4): {results.get('detailed', {}).get('win_rate_by_hand_strength', {}).get('weak', {}).get('win_rate', 0):.2%} 
         (样本数: {results.get('detailed', {}).get('win_rate_by_hand_strength', {}).get('weak', {}).get('count', 0)})
       - 中等牌 (0.4-0.7): {results.get('detailed', {}).get('win_rate_by_hand_strength', {}).get('medium', {}).get('win_rate', 0):.2%} 
         (样本数: {results.get('detailed', {}).get('win_rate_by_hand_strength', {}).get('medium', {}).get('count', 0)})
       - 强牌 (> 0.7): {results.get('detailed', {}).get('win_rate_by_hand_strength', {}).get('strong', {}).get('win_rate', 0):.2%} 
         (样本数: {results.get('detailed', {}).get('win_rate_by_hand_strength', {}).get('strong', {}).get('count', 0)})
    
    4. 按位置的胜率:
       - 主玩家: {results.get('detailed', {}).get('win_by_position', {}).get(0, 0) / args.num_games:.2%}
    
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
    # 使用game_profits的实际长度来创建x轴，确保长度匹配
    plt.plot(range(1, len(game_profits) + 1), cumulative_profits, 'b-')
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