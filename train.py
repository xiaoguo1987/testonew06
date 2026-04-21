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
import copy
import numpy as np
from datetime import datetime
import platform

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

def _resolve_auto_workers(requested_workers: int, reserved_cores: int = 2) -> int:
    """
    自动计算并行 worker 数:
    - requested_workers > 0: 直接使用
    - requested_workers <= 0: 按可用 CPU 核数自动推断
    """
    if requested_workers > 0:
        return int(requested_workers)
    logical_cores = os.cpu_count() or 4
    available = max(1, logical_cores - max(0, reserved_cores))
    # 更激进地利用 CPU：默认使用约 75% 的可用核心（小机器直接全用）
    if available <= 4:
        return available
    return max(1, (available * 3) // 4)


def _apply_hardware_profile(args) -> None:
    """
    根据硬件档位应用更激进的默认参数（仅在对应参数为“自动/未显式设置”时生效）。
    当前内置:
    - rtx3090_112cpu: 112 线程级 CPU + 单卡 3090 场景
    - rtx4090d: NVIDIA RTX 4090D (24GB VRAM) 场景，最大化GPU利用率
    """
    profile = getattr(args, 'hardware_profile', 'auto')

    if profile == 'rtx4090d':
        cpu_total = os.cpu_count() or 32
        arch = getattr(args, 'arch', 'legacy')

        if arch == 'seed-rl':
            # SEED RL GPU-Centric 架构：大模型 + 集中式 GPU 推理
            # 检测 C++ 引擎，决定 Worker/Table 配置策略
            try:
                from poker_ai.engine import USE_CC_ENGINE as _cc
            except ImportError:
                _cc = False
            if int(getattr(args, 'train_workers', 0)) <= 0:
                if _cc:
                    # C++ 引擎：减少 Pipe 数，降低 IPC 开销
                    args.train_workers = 16
                else:
                    args.train_workers = min(64, max(16, cpu_total // 4))
            if int(getattr(args, 'eval_workers', 0)) <= 0:
                args.eval_workers = min(48, max(8, cpu_total // 6))
            if int(getattr(args, 'cpu_threads', 0)) <= 0:
                args.cpu_threads = min(32, max(8, cpu_total // 4))
            if int(getattr(args, 'tf_inter_op_threads', 0)) <= 0:
                args.tf_inter_op_threads = 4
            # 大模型专用参数
            args.hidden_dims = [2048, 2048, 1024, 512]
            if int(getattr(args, 'batch_size', 64)) <= 4096:
                args.batch_size = 4096
            # 强制 ppo_epochs=4（大模型 + SEED RL 异步架构下，过多 epoch 会导致 buffer stale 引发 KL 爆炸）
            args.ppo_epochs = 4
            args.ppo_buffer_size = 16384
            args.inference_min_batch = 32
            args.inference_max_batch = 1024  # 最大推理批量（从512提升：RTX 4090D显存充足，提高GPU吞吐）
            # Anti-Fold-Bot: SEED RL 大模型需要更强的探索
            if float(getattr(args, 'ppo_entropy_coef', 0.02)) <= 0.02:
                args.ppo_entropy_coef = 0.05
            # 向量化环境：根据引擎类型决定 tables_per_worker
            if int(getattr(args, 'tables_per_worker', 0)) <= 0:
                args.tables_per_worker = 48 if _cc else 8  # C++引擎用48张表/worker（从32提升：增加GPU批量稳定性）
            # RL 训练禁用混合精度：PPO 的 log-prob 计算在 FP16 下不稳定
            args.mixed_precision = False
            # 14M 大模型学习率：使用 1e-4（折中值）
            # 3e-5 太低导致模型无法学习，3e-4 太高导致 KL 爆炸/速度崩塌
            if float(getattr(args, 'learning_rate', 3e-4)) >= 1e-4:
                args.learning_rate = 1e-4
        else:
            # Legacy 架构：CPU 多进程采样 + 小模型
            if int(getattr(args, 'train_workers', 0)) <= 0:
                args.train_workers = min(64, max(16, cpu_total // 4))
            if int(getattr(args, 'eval_workers', 0)) <= 0:
                args.eval_workers = min(48, max(8, cpu_total // 6))
            if int(getattr(args, 'cpu_threads', 0)) <= 0:
                args.cpu_threads = min(32, max(8, cpu_total // 4))
            if int(getattr(args, 'tf_inter_op_threads', 0)) <= 0:
                args.tf_inter_op_threads = 4

        # PPO 训练并行批次策略（legacy 模式使用）
        if int(getattr(args, 'train_rollouts_per_worker', 0)) <= 0:
            args.train_rollouts_per_worker = 8
        if int(getattr(args, 'train_sync_every_batches', 0)) <= 0:
            args.train_sync_every_batches = 8

        # PPO 批次大小（仅 legacy 模式回退默认值）
        if arch != 'seed-rl':
            if int(getattr(args, 'ppo_update_min_steps', 0)) <= 0:
                args.ppo_update_min_steps = 0
            if int(getattr(args, 'batch_size', 64)) <= 256:
                args.batch_size = 256
            if int(getattr(args, 'ppo_epochs', 10)) <= 10:
                args.ppo_epochs = 15

        # 默认走 GPU + XLA + 混合精度（seed-rl 已单独禁用 FP16）
        if getattr(args, 'device', 'auto') == 'auto':
            args.device = 'gpu'
        if not hasattr(args, 'mixed_precision') or arch != 'seed-rl':
            args.mixed_precision = True
        args.enable_xla = True
        args.gpu_memory_growth = True
        return

    if profile != 'rtx3090_112cpu':
        return

    cpu_total = os.cpu_count() or 112
    cpu_total = max(cpu_total, 112)

    # 线程池并发：Python牌局逻辑较重，worker 不宜直接打满 112，采用更高效区间
    if int(getattr(args, 'train_workers', 0)) <= 0:
        args.train_workers = min(48, max(24, cpu_total // 2))
    if int(getattr(args, 'eval_workers', 0)) <= 0:
        args.eval_workers = min(80, max(32, (cpu_total * 3) // 4))

    # TensorFlow CPU 并行参数（仍会在 setup_environment 阶段统一设置）
    if int(getattr(args, 'cpu_threads', 0)) <= 0:
        args.cpu_threads = min(96, cpu_total)
    if int(getattr(args, 'tf_inter_op_threads', 0)) <= 0:
        args.tf_inter_op_threads = 8

    # 训练并行批次策略：减少线程调度开销，提高 worker 有效工作占比
    if int(getattr(args, 'train_rollouts_per_worker', 0)) <= 0:
        args.train_rollouts_per_worker = 8
    if int(getattr(args, 'train_sync_every_batches', 0)) <= 0:
        args.train_sync_every_batches = 24

    # PPO 更大轨迹累计与 mini-batch，提升 3090 利用率
    if int(getattr(args, 'ppo_update_min_steps', 0)) <= 0:
        args.ppo_update_min_steps = 32768
    if int(getattr(args, 'batch_size', 64)) <= 64:
        args.batch_size = 512

    # 默认优先走 GPU+XLA+混合精度（用户显式指定时仍可覆盖）
    if getattr(args, 'device', 'auto') == 'auto':
        args.device = 'gpu'
    args.mixed_precision = True
    args.enable_xla = True
    args.gpu_memory_growth = True

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='德州扑克AI模型训练和评估脚本')
    
    # 模式选择
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'demo'],
                        default='train', help='运行模式: train(训练), eval(评估), demo(演示)')
    parser.add_argument('--hardware-profile', type=str, default='auto',
                        choices=['auto', 'rtx3090_112cpu', 'rtx4090d'],
                        help='硬件优化档位: auto, rtx3090_112cpu 或 rtx4090d')
    parser.add_argument('--arch', type=str, default='legacy',
                        choices=['legacy', 'seed-rl'],
                        help='训练架构: legacy(原有CPU采样) 或 seed-rl(GPU集中式推理+大模型)')
    parser.add_argument('--auto-tune', action='store_true',
                        help='训练前短跑多组参数，自动选择吞吐最高配置（推荐PPO）')
    parser.add_argument('--auto-tune-hands', type=int, default=4000,
                        help='自动调参每组短跑手数')
    parser.add_argument('--auto-tune-max-candidates', type=int, default=3,
                        help='自动调参最多测试候选组数')
    
    # 模型参数
    parser.add_argument('--model-type', type=str, choices=['dqn', 'policy_gradient', 'ppo'],
                        default='ppo', help='模型类型')
    parser.add_argument('--model-path', type=str, default=None,
                        help='预训练模型路径；PPO 为 base 路径（不含 .actor.weights.h5）')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='检查点保存目录')

    # 训练参数
    parser.add_argument('--num-hands', type=int, default=10000, help='训练的手牌数量')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='学习率（PPO 默认 3e-4）')
    parser.add_argument('--batch-size', type=int, default=64, help='批处理大小')
    # PPO 专用
    parser.add_argument('--ppo-clip', type=float, default=0.2, help='PPO clip_epsilon')
    parser.add_argument('--ppo-epochs', type=int, default=15, help='PPO 每批轨迹更新轮次（GPU利用率优化，KL早停保护策略稳定性）')
    parser.add_argument('--ppo-entropy-coef', type=float, default=0.02, help='PPO 熵正则系数，越大越鼓励探索')
    parser.add_argument('--ppo-update-min-steps', type=int, default=0,
                        help='PPO累计多少步再做一次更新（<=0 自动，能提升GPU吞吐）')
    parser.add_argument('--target-update', type=int, default=10, help='DQN目标网络更新频率')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamax', 'rmsprop', 'sgd'], help='优化器类型')
    parser.add_argument('--eval-interval', type=int, default=5000, help='评估间隔（局数）')
    parser.add_argument('--checkpoint-interval', type=int, default=1000, help='保存检查点间隔（局数）')
    parser.add_argument('--num-eval-games', type=int, default=100, help='每次评估的游戏局数')
    parser.add_argument('--eval-workers', type=int, default=0, help='评估并行worker数量（<=0 自动按CPU核数推断）')
    parser.add_argument('--train-workers', type=int, default=0, help='训练采样并行worker数量（<=0 自动按CPU核数推断）')
    parser.add_argument('--skip-final-eval', action='store_true', help='训练结束后跳过最终评估（加速）')
    parser.add_argument('--allin-unlucky-eq-threshold', type=float, default=0.65,
                        help='all-in 不走运补偿触发阈值（翻牌前/翻牌，若估计胜率高于该值但输掉则补偿）')
    parser.add_argument('--allin-unlucky-bonus-cap', type=float, default=0.80,
                        help='all-in 不走运补偿上限')
    parser.add_argument('--cpu-threads', type=int, default=0, help='TensorFlow intra-op 线程数（<=0 自动按CPU核数）')
    parser.add_argument('--tf-inter-op-threads', type=int, default=0, help='TensorFlow inter-op 线程数（<=0 自动）')
    parser.add_argument('--train-rollouts-per-worker', type=int, default=0,
                        help='每个训练worker单次提交的局数（<=0 自动，提升CPU利用率）')
    parser.add_argument('--train-sync-every-batches', type=int, default=0,
                        help='每隔多少并行批次同步一次权重到worker（<=0 自动）')
    parser.add_argument('--dqn-replay-updates-per-hand', type=int, default=0,
                        help='DQN每手牌回放更新次数（<=0 自动）')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'gpu', 'cpu'],
                        help='训练设备偏好: auto/gpu/cpu')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                        help='启用混合精度（GPU上通常更快）')
    parser.add_argument('--no-mixed-precision', action='store_false', dest='mixed_precision',
                        help='禁用混合精度')
    parser.add_argument('--enable-xla', action='store_true', default=True, help='启用 TensorFlow XLA JIT')
    parser.add_argument('--disable-xla', action='store_false', dest='enable_xla', help='禁用 TensorFlow XLA JIT')
    parser.add_argument('--disable-onednn', action='store_true', dest='disable_onednn', default=False,
                        help='禁用 oneDNN（可避免部分 GPU 上 DNN library initialization failed）')
    parser.add_argument('--gpu-memory-growth', action='store_true', default=True,
                        help='启用GPU按需分配显存')
    parser.add_argument('--no-gpu-memory-growth', action='store_false', dest='gpu_memory_growth',
                        help='禁用GPU按需分配显存')
    
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
    parser.add_argument('--opponent-update-interval', type=int, default=10000, help='对手更新间隔（局数），过小会导致军备竞赛')
    
    # 探索参数
    parser.add_argument('--eps-start', type=float, default=1.0, help='ε-贪婪策略初始探索率')
    parser.add_argument('--eps-end', type=float, default=0.01, help='ε-贪婪策略最终探索率')
    parser.add_argument('--eps-decay', type=float, default=0.0001, help='ε衰减速率')
    
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
    
    args.train_workers = _resolve_auto_workers(getattr(args, 'train_workers', 0), reserved_cores=2)
    args.eval_workers = _resolve_auto_workers(getattr(args, 'eval_workers', 0), reserved_cores=1)

    # 记录硬件使用情况与并行配置
    try:
        import tensorflow as tf
        from tensorflow.keras import mixed_precision
        logical_cores = os.cpu_count()
        if hasattr(os, "sched_getaffinity"):
            available_cores = len(os.sched_getaffinity(0))
        else:
            available_cores = logical_cores

        # 线程并行配置（在创建图/执行前设置）
        requested_intra = int(getattr(args, 'cpu_threads', 0))
        requested_inter = int(getattr(args, 'tf_inter_op_threads', 0))
        intra_threads = requested_intra if requested_intra > 0 else max(1, int(available_cores))
        inter_threads = (
            requested_inter if requested_inter > 0
            else max(1, min(8, intra_threads // 4))
        )
        tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter_threads)

        gpus = tf.config.list_physical_devices('GPU')
        cpus = tf.config.list_physical_devices('CPU')
        has_gpu = len(gpus) > 0
        if args.device == 'cpu':
            try:
                tf.config.set_visible_devices([], 'GPU')
                gpus = []
                has_gpu = False
            except Exception as _e:
                logger.warning(f"切换到CPU模式失败，将继续使用默认设备: {_e}")
        elif args.device == 'gpu' and not has_gpu:
            logger.warning("指定了 --device gpu，但未检测到可用GPU，将回退到CPU")

        if has_gpu and getattr(args, 'gpu_memory_growth', True):
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception as _e:
                    logger.warning(f"设置GPU显存按需增长失败({gpu.name}): {_e}")

        if has_gpu and getattr(args, 'mixed_precision', True):
            mixed_precision.set_global_policy('mixed_float16')
            logger.info("已启用 mixed precision: mixed_float16")
        else:
            mixed_precision.set_global_policy('float32')
            logger.info("mixed precision 已禁用，使用 float32")

        if getattr(args, 'enable_xla', True):
            tf.config.optimizer.set_jit(True)
            logger.info("已启用 XLA JIT")
        else:
            tf.config.optimizer.set_jit(False)
            logger.info("XLA JIT 已禁用")

        logger.info(f"TensorFlow版本: {tf.__version__}")
        logger.info(
            f"DEBUG: 平台={platform.system()}, 逻辑核心数={logical_cores}, "
            f"进程可用核心数={available_cores}"
        )
        logger.info(f"并行配置: TF intra_op={intra_threads}, inter_op={inter_threads}")
        logger.info(f"并行配置: train_workers={args.train_workers}, eval_workers={args.eval_workers}")
        try:
            from poker_ai.engine import USE_CC_ENGINE as _cc_flag
        except ImportError:
            _cc_flag = False
        logger.info(f"引擎状态: {'C++ Engine' if _cc_flag else 'Python Engine'}, "
                    f"tables_per_worker={getattr(args, 'tables_per_worker', 8)}")
        logger.info(
            f"并行配置: train_rollouts_per_worker={getattr(args, 'train_rollouts_per_worker', 0)}, "
            f"train_sync_every_batches={getattr(args, 'train_sync_every_batches', 0)}"
        )
        logger.info(
            f"PPO配置: batch_size={getattr(args, 'batch_size', 64)}, "
            f"ppo_update_min_steps={getattr(args, 'ppo_update_min_steps', 0)}"
        )
        logger.info(
            f"训练策略: dynamic_opp_interval(Phase1=50K/Phase2=50K/Phase3=150K), "
            f"critic_lr_multiplier=3.0, Curriculum=Phase1(全鱼)→Phase2(混合)→Phase3(自对弈), "
            f"历史对手池=开启(pool_size=20, 70%近期+30%远期), VPIP奖励=已禁用"
        )
        logger.info(f"可用GPU数量: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            logger.info(f"GPU {i}: {gpu.name}")
        logger.info(f"TF CPU设备数: {len(cpus)}（这是TensorFlow逻辑设备数，不等于CPU核心数）")
        logger.info(f"训练将使用: {'GPU' if gpus else 'CPU'}")

        # GPU 预热验证：确认 GPU 计算确实可用
        if has_gpu:
            try:
                with tf.device('/GPU:0'):
                    _a = tf.random.normal([5000, 5000])
                    _b = tf.random.normal([5000, 5000])
                    _c = tf.matmul(_a, _b)
                    # 强制执行并获取设备信息
                    _val = _c[0, 0].numpy()
                    logger.info(f"[GPU诊断] GPU预热矩阵乘法成功，结果设备: {_c.device}")
                    del _a, _b, _c, _val
            except Exception as _e:
                logger.warning(f"[GPU诊断] GPU预热失败，可能回退到CPU: {_e}")
    except ImportError:
        logger.warning("TensorFlow未安装，无法检测硬件信息")
    except Exception as e:
        logger.warning(f"检测硬件信息时出错: {e}")
    
    logger.info(f"环境设置完成: 检查点目录={args.checkpoint_dir}")
    logger.info(f"当前运行模式: {args.mode}")
    logger.info(f"使用模型类型: {args.model_type}")

def _train_seed_rl(args):
    """SEED RL GPU-Centric 训练入口。"""
    logger.info("[SEED RL] 启动 GPU-Centric 训练架构")
    logger.info(f"训练参数: {vars(args)}")
    start_time = time.time()

    try:
        from poker_ai.model.large_ppo_model import LargePPOModel
        from poker_ai.training.gpu_trainer import SeedRLTrainer
        from poker_ai.config.ppo_config import PPOConfig

        # 构建大模型配置
        hidden_dims = getattr(args, 'hidden_dims', [2048, 2048, 1024, 512])
        cfg = PPOConfig(
            lr=args.learning_rate,
            batch_size=args.batch_size,
            update_epochs=getattr(args, 'ppo_epochs', 10),
            clip_epsilon=getattr(args, 'ppo_clip', 0.2),
            entropy_coef=getattr(args, 'ppo_entropy_coef', 0.02),
            hidden_dims=hidden_dims,
        )

        model = LargePPOModel(
            hidden_dims=hidden_dims,
            batch_size=args.batch_size,
            update_epochs=getattr(args, 'ppo_epochs', 10),
            config=cfg,
        )

        trainer = SeedRLTrainer(
            model=model,
            checkpoint_dir=args.checkpoint_dir,
            small_blind=args.small_blind,
            big_blind=args.big_blind,
            initial_stack=args.initial_stack,
            num_workers=args.train_workers,
            eval_interval=args.eval_interval,
            checkpoint_interval=args.checkpoint_interval,
            num_eval_games=args.num_eval_games,
            opponent_update_interval=args.opponent_update_interval,
            inference_min_batch=getattr(args, 'inference_min_batch', 32),
            inference_max_batch=getattr(args, 'inference_max_batch', 512),
            ppo_buffer_size=getattr(args, 'ppo_buffer_size', 4096),
            skip_final_eval=getattr(args, 'skip_final_eval', False),
            allin_eq_threshold=getattr(args, 'allin_unlucky_eq_threshold', 0.65),
            allin_bonus_cap=getattr(args, 'allin_unlucky_bonus_cap', 0.80),
            tables_per_worker=getattr(args, 'tables_per_worker', 8),
        )

        # 加载 checkpoint
        if args.model_path:
            trainer.load_checkpoint(args.model_path)
        elif args.resume:
            trainer.load_checkpoint()

        logger.info(f"[SEED RL] 开始训练 {args.num_hands} 手牌...")
        trainer.train(args.num_hands)

        elapsed = time.time() - start_time
        logger.info(f"[SEED RL] 训练完成！总耗时: {elapsed:.2f}s")
        save_training_summary(args, elapsed)
        return True

    except Exception as e:
        logger.error(f"[SEED RL] 训练出错: {e}")
        raise


def train_model(args):
    """
    训练模型（Legacy 架构）
    """
    # 如果选择了 seed-rl 架构，走专用入口
    if getattr(args, 'arch', 'legacy') == 'seed-rl':
        return _train_seed_rl(args)

    logger.info("开始训练德州扑克AI模型")
    logger.info(f"训练参数: {vars(args)}")
    
    start_time = time.time()
    
    try:
        from poker_ai.reward.self_play_trainer import SelfPlayTrainer
        
        def _create_trainer(override: dict = None, tuning_mode: bool = False):
            ov = override or {}
            return SelfPlayTrainer(
                model_type=args.model_type,
                checkpoint_dir=args.checkpoint_dir,
                small_blind=args.small_blind,
                big_blind=args.big_blind,
                initial_stack=args.initial_stack,
                num_players=args.num_players,
                eval_interval=ov.get('eval_interval', args.eval_interval),
                checkpoint_interval=ov.get('checkpoint_interval', args.checkpoint_interval),
                num_eval_games=ov.get('num_eval_games', args.num_eval_games),
                use_ensemble=args.use_ensemble,
                opponent_update_interval=ov.get('opponent_update_interval', args.opponent_update_interval),
                learning_rate=args.learning_rate,
                batch_size=ov.get('batch_size', args.batch_size),
                optimizer=args.optimizer,
                target_update_freq=args.target_update,
                ppo_clip_epsilon=getattr(args, 'ppo_clip', 0.2),
                ppo_update_epochs=getattr(args, 'ppo_epochs', 10),
                ppo_entropy_coef=getattr(args, 'ppo_entropy_coef', 0.02),
                ppo_update_min_steps=ov.get('ppo_update_min_steps', getattr(args, 'ppo_update_min_steps', 0)),
                train_rollouts_per_worker=ov.get('train_rollouts_per_worker', getattr(args, 'train_rollouts_per_worker', 0)),
                train_sync_every_batches=ov.get('train_sync_every_batches', getattr(args, 'train_sync_every_batches', 0)),
                dqn_replay_updates_per_hand=ov.get('dqn_replay_updates_per_hand', getattr(args, 'dqn_replay_updates_per_hand', 0)),
                tuning_mode=tuning_mode,
                skip_final_eval=getattr(args, 'skip_final_eval', False),
                allin_unlucky_eq_threshold=getattr(args, 'allin_unlucky_eq_threshold', 0.65),
                allin_unlucky_bonus_cap=getattr(args, 'allin_unlucky_bonus_cap', 0.80),
                eval_workers=ov.get('eval_workers', getattr(args, 'eval_workers', 1)),
                train_workers=ov.get('train_workers', getattr(args, 'train_workers', 1)),
            )

        def _configure_exploration(trainer_obj):
            if hasattr(trainer_obj, 'main_agent') and trainer_obj.main_agent:
                if hasattr(trainer_obj.main_agent, 'set_epsilon'):
                    trainer_obj.main_agent.set_epsilon(
                        value=args.eps_start,
                        min_value=args.eps_end,
                        decay_rate=args.eps_decay
                    )
                elif hasattr(trainer_obj.main_agent, 'agent'):
                    if hasattr(trainer_obj.main_agent.agent, 'epsilon'):
                        trainer_obj.main_agent.agent.epsilon = args.eps_start
                    if hasattr(trainer_obj.main_agent.agent, 'epsilon_min'):
                        trainer_obj.main_agent.agent.epsilon_min = args.eps_end
                    if hasattr(trainer_obj.main_agent.agent, 'epsilon_decay'):
                        trainer_obj.main_agent.agent.epsilon_decay = args.eps_decay
                    logger.info(
                        f"已通过直接访问设置探索参数: eps_start={args.eps_start}, "
                        f"eps_end={args.eps_end}, eps_decay={args.eps_decay}"
                    )
            if hasattr(trainer_obj, 'opponent_update_interval'):
                trainer_obj.opponent_update_interval = args.opponent_update_interval

        # 自动调参：短跑多组候选，按 hands/s 选最优
        if getattr(args, 'auto_tune', False) and args.model_type == 'ppo':
            cpu_total = os.cpu_count() or 32
            candidate_pool = [
                {
                    'train_workers': max(8, min(getattr(args, 'train_workers', 16), 24)),
                    'train_rollouts_per_worker': max(4, int(getattr(args, 'train_rollouts_per_worker', 4))),
                    'train_sync_every_batches': max(8, int(getattr(args, 'train_sync_every_batches', 8))),
                    'batch_size': max(256, int(getattr(args, 'batch_size', 256))),
                    'ppo_update_min_steps': max(8192, int(getattr(args, 'ppo_update_min_steps', 8192))),
                },
                {
                    'train_workers': max(16, min(cpu_total // 3, 36)),
                    'train_rollouts_per_worker': 8,
                    'train_sync_every_batches': 16,
                    'batch_size': 512,
                    'ppo_update_min_steps': 16384,
                },
                {
                    'train_workers': max(24, min(cpu_total // 2, 48)),
                    'train_rollouts_per_worker': 8,
                    'train_sync_every_batches': 24,
                    'batch_size': 512,
                    'ppo_update_min_steps': 32768,
                },
                {
                    'train_workers': max(32, min((cpu_total * 2) // 3, 64)),
                    'train_rollouts_per_worker': 12,
                    'train_sync_every_batches': 32,
                    'batch_size': 768,
                    'ppo_update_min_steps': 65536,
                },
            ]

            max_candidates = max(1, int(getattr(args, 'auto_tune_max_candidates', 3)))
            candidates = candidate_pool[:max_candidates]
            warmup_hands = max(500, int(getattr(args, 'auto_tune_hands', 4000)))
            # 小规模测试时，自动调参不应吞掉大部分预算
            if args.num_hands <= warmup_hands * 2:
                warmup_hands = max(200, args.num_hands // 4)
                logger.info(
                    f"[AutoTune] 检测到小规模训练(num_hands={args.num_hands})，"
                    f"自动将 warmup_hands 调整为 {warmup_hands}"
                )
            logger.info(f"自动调参启动: candidates={len(candidates)}, warmup_hands={warmup_hands}")

            best_idx = -1
            best_hps = -1.0
            for i, cand in enumerate(candidates, start=1):
                logger.info(f"[AutoTune] 候选 {i}/{len(candidates)}: {cand}")
                tune_override = copy.deepcopy(cand)
                tune_override['eval_interval'] = 10**9
                tune_override['checkpoint_interval'] = 10**9
                tune_override['opponent_update_interval'] = 10**9
                tune_override['num_eval_games'] = 1
                t0 = time.perf_counter()
                tune_trainer = _create_trainer(override=tune_override, tuning_mode=True)
                _configure_exploration(tune_trainer)
                tune_trainer.train(warmup_hands)
                elapsed = max(1e-6, time.perf_counter() - t0)
                hps = warmup_hands / elapsed
                logger.info(
                    f"[AutoTune] 候选{i}结果: hands/s={hps:.2f}, elapsed={elapsed:.2f}s"
                )
                if hps > best_hps:
                    best_hps = hps
                    best_idx = i - 1

            best = candidates[best_idx]
            args.train_workers = best['train_workers']
            args.train_rollouts_per_worker = best['train_rollouts_per_worker']
            args.train_sync_every_batches = best['train_sync_every_batches']
            args.batch_size = best['batch_size']
            args.ppo_update_min_steps = best['ppo_update_min_steps']
            logger.info(f"[AutoTune] 选择最优候选: #{best_idx + 1}, hands/s={best_hps:.2f}, config={best}")

        # 创建正式训练器
        trainer = _create_trainer()
        
        # 如果启用详细日志，更新日志级别
        if args.verbose:
            logger.setLevel(logging.DEBUG)
            
        _configure_exploration(trainer)
        
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
        
        # 添加训练进度日志
        start_hand = trainer.total_hands if hasattr(trainer, 'total_hands') else 0
        logger.info(f"初始手牌数: {start_hand}, 目标手牌数: {start_hand + args.num_hands}")
        
        # 开始训练过程
        trainer.train(args.num_hands)
        
        # 计算训练时间
        end_time = time.time()
        training_time = end_time - start_time
        
        # 计算实际训练的手牌数
        if hasattr(trainer, 'total_hands'):
            actual_hands = trainer.total_hands - start_hand
            logger.info(f"实际训练手牌数: {actual_hands}")
        
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
    
    from poker_ai.reward.self_play_trainer import SelfPlayTrainer
    trainer = SelfPlayTrainer(
        model_type=args.model_type,
        checkpoint_dir=args.checkpoint_dir,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        initial_stack=args.initial_stack,
        num_players=args.num_players,
        num_eval_games=args.num_eval_games,
        target_update_freq=args.target_update,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        ppo_clip_epsilon=getattr(args, 'ppo_clip', 0.2),
        ppo_update_epochs=getattr(args, 'ppo_epochs', 10),
        ppo_entropy_coef=getattr(args, 'ppo_entropy_coef', 0.02),
        ppo_update_min_steps=getattr(args, 'ppo_update_min_steps', 0),
        train_rollouts_per_worker=getattr(args, 'train_rollouts_per_worker', 0),
        train_sync_every_batches=getattr(args, 'train_sync_every_batches', 0),
        dqn_replay_updates_per_hand=getattr(args, 'dqn_replay_updates_per_hand', 0),
        eval_workers=getattr(args, 'eval_workers', 1),
        train_workers=getattr(args, 'train_workers', 1),
    )
    
    # 加载模型
    if args.model_path:
        logger.info(f"使用指定的模型路径: {args.model_path}")
        success = trainer.load_checkpoint(args.model_path)
    else:
        logger.info("未指定模型路径，尝试加载最近的检查点...")
        success = trainer.load_checkpoint()
    
    if not success:
        logger.error("无法加载模型，请使用 --model-path 参数指定有效的模型路径")
        return False
    
    try:
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
        from poker_ai.advisor.ai_advisor import AIAvisor
        from poker_ai.game.poker_game import PokerGame, Player
        from poker_ai.model.ai_agent import PokerAIAgent
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
        _apply_hardware_profile(args)

        # 在 import TensorFlow 前禁用 oneDNN，避免部分环境 DNN library initialization failed
        if getattr(args, 'disable_onednn', False):
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
            logger.info("已设置 TF_ENABLE_ONEDNN_OPTS=0（禁用 oneDNN）")

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