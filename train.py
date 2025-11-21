#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
德州扑克AI模型训练和评估脚本

此脚本提供命令行接口，用于训练和评估德州扑克AI模型。
支持多种参数配置，包括模型类型、训练轮数、学习率等。
"""

import os
import argparse
import json
import logging
import time
import numpy as np
from datetime import datetime

# 设置日志目录和文件名
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"poker_ai_{timestamp}.log")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # 日志文件
        logging.StreamHandler()         # 控制台输出
    ]
)
logger = logging.getLogger('PokerAITrainer')
logger.info(f"日志已配置，记录到: {log_file}")

# 导入必要的模块
try:
    from poker_ai.reward.self_play_trainer import SelfPlayTrainer
    from poker_ai.advisor.ai_advisor import AIAvisor
    from poker_ai.game.poker_game import PokerGame, Player
    from poker_ai.model.ai_agent import PokerAIAgent
    logger.info("成功导入所有必要模块")
except Exception as e:
    logger.error(f"导入模块失败: {e}")
    raise

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='德州扑克AI模型训练和评估脚本')
    
    # 模式选择
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'demo'],
                        default='train', help='运行模式: train(训练), eval(评估), demo(演示)')
    
    # 模型参数
    parser.add_argument('--model-type', type=str, choices=['dqn', 'policy_gradient'],
                        default='dqn', help='模型类型')
    parser.add_argument('--model-path', type=str, default=None,
                        help='预训练模型路径，用于继续训练或评估')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='检查点保存目录')
    
    # 训练参数
    parser.add_argument('--num-hands', type=int, default=10000, help='训练的手牌数量')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--batch-size', type=int, default=64, help='批处理大小')
    parser.add_argument('--eval-interval', type=int, default=100, help='评估间隔（局数）')
    parser.add_argument('--checkpoint-interval', type=int, default=500, help='保存检查点间隔（局数）')
    parser.add_argument('--num-eval-games', type=int, default=50, help='每次评估的游戏局数')
    
    # 游戏参数
    parser.add_argument('--small-blind', type=int, default=5, help='小盲注')
    parser.add_argument('--big-blind', type=int, default=10, help='大盲注')
    parser.add_argument('--initial-stack', type=int, default=1000, help='初始筹码量')
    parser.add_argument('--num-players', type=int, default=6, help='玩家数量')
    
    # 高级参数
    parser.add_argument('--use-ensemble', action='store_true', help='使用集成模型')
    parser.add_argument('--resume', action='store_true', help='从最近的检查点继续训练')
    parser.add_argument('--save-best-only', action='store_true', help='仅保存最佳模型')
    parser.add_argument('--verbose', action='store_true', help='输出详细日志')
    
    # 评估参数
    parser.add_argument('--eval-opponent', type=str, default='previous_version',
                        help='评估对手类型: previous_version, random, baseline')
    parser.add_argument('--eval-output', type=str, default='eval_results.json',
                        help='评估结果输出文件')
    
    # 演示参数
    parser.add_argument('--demo-hands', type=int, default=10, help='演示的手牌数量')
    
    # 自我对弈参数
    parser.add_argument('--use-self-play', action='store_true', default=True, help='启用自我对弈训练')
    parser.add_argument('--opponent-update-interval', type=int, default=10000, help='对手更新间隔（局数）')
    
    # 探索参数
    parser.add_argument('--eps-start', type=float, default=1.0, help='ε-贪婪策略初始探索率')
    parser.add_argument('--eps-end', type=float, default=0.01, help='ε-贪婪策略最终探索率')
    parser.add_argument('--eps-decay', type=float, default=0.0001, help='ε衰减速率')
    parser.add_argument('--target-update', type=int, default=10, help='目标网络更新频率')
    
    return parser.parse_args()

def setup_environment(args):
    """
    设置训练环境
    """
    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 如果启用详细日志，更新日志级别
    if args.verbose:
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    logger.info(f"环境设置完成: 检查点目录={args.checkpoint_dir}")
    logger.info(f"当前运行模式: {args.mode}")
    logger.info(f"使用模型类型: {args.model_type}")

def train_model(args):
    """
    训练模型
    """
    logger.info("开始训练德州扑克AI模型")
    logger.info(f"训练参数: {vars(args)}")
    
    start_time = time.time()
    
    try:
        # 创建训练器
        trainer = SelfPlayTrainer(
            model_type=args.model_type,
            checkpoint_dir=args.checkpoint_dir,
            small_blind=args.small_blind,
            big_blind=args.big_blind,
            initial_stack=args.initial_stack,
            num_players=args.num_players,
            eval_interval=args.eval_interval,
            checkpoint_interval=args.checkpoint_interval,
            num_eval_games=args.num_eval_games,
            use_ensemble=args.use_ensemble,
            opponent_update_interval=args.opponent_update_interval
        )
        
        # 设置探索参数
        if hasattr(trainer, 'main_agent') and trainer.main_agent:
            # 直接设置探索率（如果是DQN代理）
            if hasattr(trainer.main_agent, 'agent') and hasattr(trainer.main_agent.agent, 'epsilon'):
                trainer.main_agent.agent.epsilon = args.eps_start
            if hasattr(trainer.main_agent, 'agent') and hasattr(trainer.main_agent.agent, 'epsilon_min'):
                trainer.main_agent.agent.epsilon_min = args.eps_end
            # 设置衰减率
            if hasattr(trainer.main_agent, 'agent') and hasattr(trainer.main_agent.agent, 'epsilon_decay'):
                trainer.main_agent.agent.epsilon_decay = args.eps_decay
        
        # 设置对手更新间隔
        if hasattr(trainer, 'opponent_update_interval'):
            trainer.opponent_update_interval = args.opponent_update_interval
        
        # 如果指定了模型路径或需要恢复训练
        if args.model_path:
            success = trainer.load_checkpoint(args.model_path)
            if not success:
                logger.warning(f"无法加载指定的模型: {args.model_path}")
        elif args.resume:
            success = trainer.load_checkpoint()
            if not success:
                logger.warning("没有找到可恢复的检查点，将从头开始训练")
        
        # 开始训练
        logger.info(f"开始训练 {args.num_hands} 手牌...")
        trainer.train(args.num_hands)
        
        # 计算训练时间
        end_time = time.time()
        training_time = end_time - start_time
        
        logger.info(f"训练完成！总耗时: {training_time:.2f} 秒")
        logger.info(f"平均每手耗时: {training_time / args.num_hands:.4f} 秒")
        
        # 保存训练总结
        save_training_summary(args, training_time)
        
        return True
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        raise

def evaluate_model(args):
    """
    评估模型
    """
    logger.info("开始评估德州扑克AI模型")
    
    if not args.model_path:
        logger.error("评估模式需要指定模型路径，请使用 --model-path 参数")
        return False
    
    try:
        # 创建训练器用于评估
        trainer = SelfPlayTrainer(
            model_type=args.model_type,
            checkpoint_dir=args.checkpoint_dir,
            small_blind=args.small_blind,
            big_blind=args.big_blind,
            initial_stack=args.initial_stack,
            num_players=args.num_players,
            num_eval_games=args.num_eval_games
        )
        
        # 加载模型
        success = trainer.load_checkpoint(args.model_path)
        if not success:
            logger.error(f"无法加载模型: {args.model_path}")
            return False
        
        # 执行评估
        logger.info(f"执行评估，共进行 {args.num_eval_games} 局游戏...")
        eval_results = trainer.evaluate()
        
        # 记录评估结果
        logger.info(f"评估结果: {eval_results}")
        
        # 保存评估结果到文件
        save_eval_results(args, eval_results)
        
        # 生成评估报告
        generate_eval_report(args, eval_results)
        
        return True
    except Exception as e:
        logger.error(f"评估过程中发生错误: {e}")
        raise

def run_demo(args):
    """
    演示模型
    """
    logger.info("开始演示德州扑克AI模型")
    
    if not args.model_path:
        logger.error("演示模式需要指定模型路径，请使用 --model-path 参数")
        return False
    
    try:
        # 创建AI代理
        agent = PokerAIAgent(agent_type=args.model_type)
        
        # 加载模型
        success = agent.load_model(args.model_path)
        if not success:
            logger.error(f"无法加载模型: {args.model_path}")
            return False
        
        # 创建游戏环境
        players = [Player(f"Player_{i+1}", args.initial_stack, i == 0) for i in range(args.num_players)]
        game = PokerGame(args.small_blind, args.big_blind, players)
        
        # 创建AI建议器
        advisor = AIAvisor(model_path=args.model_path, model_type=args.model_type)
        
        # 运行演示
        logger.info(f"开始演示 {args.demo_hands} 手牌...")
        demo_results = []
        
        for hand_idx in range(args.demo_hands):
            logger.info(f"\n--- 第 {hand_idx + 1} 手牌 ---")
            
            # 开始新的一手牌
            game.start_new_hand()
            
            # 打印初始状态
            print(f"\n开始新的一手牌: 第 {hand_idx + 1} 手")
            print(f"盲注: 小盲={args.small_blind}, 大盲={args.big_blind}")
            print("玩家手牌:")
            for player in game.players:
                if player.is_human:
                    print(f"  {player.name}: {[str(card) for card in player.cards]}")
                else:
                    print(f"  {player.name}: [隐藏]")
            
            # 游戏主循环
            while True:
                # 处理当前轮次
                while not game.is_round_over():
                    current_player = game.next_player()
                    if not current_player:
                        break
                    
                    print(f"\n轮到 {current_player.name} 行动")
                    print(f"当前底池: {game.pot}")
                    print(f"当前下注: {game.current_bet}")
                    
                    if current_player.is_human:
                        # 获取AI建议
                        advice = advisor.get_basic_advice(game, current_player)
                        print(f"\nAI建议:")
                        print(f"  手牌强度: {advice['hand_strength']:.2%}")
                        print(f"  手牌类型: {advice['hand_type']}")
                        print(f"  推荐动作: {advice['recommended_action']}")
                        print(f"  推荐金额: {advice['recommended_amount']}")
                        print(f"  预期价值: {advice['expected_value']:.2f}")
                        print(f"  置信度: {advice['confidence']:.2%}")
                        print(f"  理由: {advice['reasoning']}")
                    
                    # AI代理选择动作
                    action, amount = agent.select_action(game, current_player)
                    
                    # 记录动作
                    print(f"{current_player.name} 决定: {action}")
                    if amount > 0:
                        print(f"金额: {amount}")
                    
                    # 执行动作
                    game.handle_action(current_player, action, amount)
                
                # 显示公共牌
                if game.community_cards:
                    print(f"\n公共牌: {[str(card) for card in game.community_cards]}")
                
                # 进入下一轮或结束
                if game.proceed_to_next_round():
                    break
            
            # 显示结果
            print(f"\n--- 手牌结果 ---")
            print(f"获胜者: {game.winner.name if game.winner else '无'}")
            print("最终筹码:")
            for player in game.players:
                print(f"  {player.name}: {player.stack}")
            
            # 记录结果
            demo_results.append({
                'hand_index': hand_idx + 1,
                'winner': game.winner.name if game.winner else '无',
                'final_stacks': {p.name: p.stack for p in game.players}
            })
        
        # 保存演示结果
        save_demo_results(args, demo_results)
        
        logger.info("演示完成！")
        return True
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        raise

def save_training_summary(args, training_time):
    """
    保存训练总结
    """
    summary = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'training_time_seconds': training_time,
        'training_time_hours': training_time / 3600,
        'hands_per_second': args.num_hands / training_time if training_time > 0 else 0
    }
    
    # 保存到文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(args.checkpoint_dir, f"training_summary_{timestamp}.json")
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"训练总结已保存到: {filename}")
    except Exception as e:
        logger.error(f"保存训练总结失败: {e}")

def save_eval_results(args, results):
    """
    保存评估结果
    """
    eval_data = {
        'timestamp': datetime.now().isoformat(),
        'model_path': args.model_path,
        'args': vars(args),
        'results': results
    }
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.eval_output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(args.eval_output, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)
        logger.info(f"评估结果已保存到: {args.eval_output}")
    except Exception as e:
        logger.error(f"保存评估结果失败: {e}")

def generate_eval_report(args, results):
    """
    生成评估报告
    """
    report = f"""
    ======================================================
                    德州扑克AI模型评估报告
    ======================================================
    评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    模型路径: {args.model_path}
    模型类型: {args.model_type}
    评估游戏数: {args.num_eval_games}
    
    评估结果:
    - 平均利润: {results['avg_profit']:.2f}
    - 胜率: {results['win_rate']:.2%}
    - 总利润: {results['total_profit']:.2f}
    - 获胜次数: {results['wins']}
    - 总游戏数: {results['total_hands']}
    ======================================================
    """
    
    print(report)
    
    # 保存报告到文件
    report_file = args.eval_output.replace('.json', '.txt')
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"评估报告已保存到: {report_file}")
    except Exception as e:
        logger.error(f"保存评估报告失败: {e}")

def save_demo_results(args, results):
    """
    保存演示结果
    """
    demo_data = {
        'timestamp': datetime.now().isoformat(),
        'model_path': args.model_path,
        'args': vars(args),
        'results': results
    }
    
    # 保存到文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"demo_results_{timestamp}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(demo_data, f, ensure_ascii=False, indent=2)
        logger.info(f"演示结果已保存到: {filename}")
    except Exception as e:
        logger.error(f"保存演示结果失败: {e}")

def main():
    """
    主函数
    """
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 设置环境
        setup_environment(args)
        
        # 根据模式执行不同的功能
        if args.mode == 'train':
            success = train_model(args)
        elif args.mode == 'eval':
            success = evaluate_model(args)
        elif args.mode == 'demo':
            success = run_demo(args)
        else:
            logger.error(f"未知的模式: {args.mode}")
            success = False
        
        if success:
            logger.info("任务执行成功！")
        else:
            logger.error("任务执行失败！")
            return 1
        
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