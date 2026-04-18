#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PPO优化验证测试脚本

用于验证PPO模型的优化是否正常工作
"""

import os
import sys
import numpy as np
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PPOOptimizationTest')

def test_network_architecture():
    """测试网络架构改进"""
    logger.info("=" * 60)
    logger.info("测试1: 网络架构改进")
    logger.info("=" * 60)
    
    try:
        from poker_ai.model.ppo_model import PPOModel
        from poker_ai.config.ppo_config import DEFAULT_PPO_CONFIG
        
        # 创建模型
        model = PPOModel(
            input_dim=134,
            output_dim=5,
            lr=3e-4,
            config=DEFAULT_PPO_CONFIG
        )
        
        # 测试前向传播
        dummy_state = np.random.randn(1, 134).astype(np.float32)
        action_probs, value = model.predict(dummy_state)
        
        logger.info(f"✓ 模型创建成功")
        logger.info(f"  Actor输出形状: {action_probs.shape}")
        logger.info(f"  Critic输出形状: {value.shape}")
        logger.info(f"  动作概率和: {action_probs.sum():.4f}")
        logger.info(f"  状态价值: {value:.4f}")
        
        # 测试批量推理
        batch_states = np.random.randn(64, 134).astype(np.float32)
        batch_probs, batch_values = model.predict_batch(batch_states)
        
        logger.info(f"✓ 批量推理成功")
        logger.info(f"  批量形状: {batch_probs.shape}, {batch_values.shape}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 网络架构测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lr_scheduler():
    """测试学习率调度"""
    logger.info("=" * 60)
    logger.info("测试2: 学习率调度（Warmup + Cosine衰减）")
    logger.info("=" * 60)
    
    try:
        from poker_ai.model.ppo_model import PPOModel
        from poker_ai.config.ppo_config import DEFAULT_PPO_CONFIG
        
        model = PPOModel(config=DEFAULT_PPO_CONFIG)
        
        # 测试不同步数的学习率
        test_steps = [0, 500, 1000, 5000, 50000, 99000]
        logger.info("学习率调度测试:")
        
        for step in test_steps:
            lr = model._get_lr_with_warmup_and_decay(step)
            logger.info(f"  步数 {step:>6}: 学习率 = {lr:.6e}")
        
        # 验证warmup
        lr_0 = model._get_lr_with_warmup_and_decay(0)
        lr_500 = model._get_lr_with_warmup_and_decay(500)
        lr_1000 = model._get_lr_with_warmup_and_decay(1000)
        
        assert lr_0 >= model.lr_min, "初始学习率应该 >= lr_min"
        assert lr_500 > lr_0, "Warmup阶段学习率应该递增"
        assert lr_1000 >= model.initial_lr * 0.99, "Warmup结束时学习率应该接近initial_lr"
        
        logger.info(f"✓ 学习率调度测试通过")
        return True
    except Exception as e:
        logger.error(f"✗ 学习率调度测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gae_computation():
    """测试GAE计算"""
    logger.info("=" * 60)
    logger.info("测试3: GAE计算改进")
    logger.info("=" * 60)
    
    try:
        from poker_ai.model.ppo_model import PPOModel
        from poker_ai.config.ppo_config import DEFAULT_PPO_CONFIG
        
        model = PPOModel(config=DEFAULT_PPO_CONFIG)
        
        # 测试正常情况
        rewards = [1.0, 0.5, -0.5, 1.0, 0.0]
        values = [0.8, 0.6, 0.4, 0.7, 0.3]
        dones = [False, False, False, False, True]
        
        advantages = model._compute_gae(rewards, values, dones)
        
        logger.info(f"✓ GAE计算成功")
        logger.info(f"  优势值: {advantages}")
        logger.info(f"  均值: {advantages.mean():.4f}")
        logger.info(f"  标准差: {advantages.std():.4f}")
        logger.info(f"  最小值: {advantages.min():.4f}")
        logger.info(f"  最大值: {advantages.max():.4f}")
        
        # 验证数值稳定性
        assert not np.any(np.isnan(advantages)), "优势值不应包含NaN"
        assert not np.any(np.isinf(advantages)), "优势值不应包含Inf"
        assert np.all(advantages >= -10) and np.all(advantages <= 10), "优势值应在[-10, 10]范围内"
        
        logger.info(f"✓ 数值稳定性验证通过")
        return True
    except Exception as e:
        logger.error(f"✗ GAE计算测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step():
    """测试训练步骤"""
    logger.info("=" * 60)
    logger.info("测试4: PPO训练步骤")
    logger.info("=" * 60)
    
    try:
        from poker_ai.model.ppo_model import PPOModel
        from poker_ai.config.ppo_config import DEFAULT_PPO_CONFIG
        
        model = PPOModel(
            input_dim=134,
            output_dim=5,
            batch_size=32,
            update_epochs=2,
            config=DEFAULT_PPO_CONFIG
        )
        
        # 准备测试数据
        num_samples = 64
        states = np.random.randn(num_samples, 134).astype(np.float32)
        actions = np.random.randint(0, 5, size=(num_samples,))
        rewards = np.random.randn(num_samples).astype(np.float32)
        dones = np.random.choice([True, False], size=(num_samples,))
        old_probs = np.random.dirichlet(np.ones(5), size=num_samples).astype(np.float32)
        old_probs = old_probs[np.arange(num_samples), actions]
        
        # 执行训练
        loss = model.train(states, actions, rewards, dones, old_probs)
        
        logger.info(f"✓ 训练步骤成功")
        logger.info(f"  损失值: {loss:.6f}")
        logger.info(f"  当前学习率: {model.current_lr:.6e}")
        logger.info(f"  训练步数: {model.current_step}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 训练步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_entropy_adaptive():
    """测试自适应熵系数"""
    logger.info("=" * 60)
    logger.info("测试5: 自适应熵系数")
    logger.info("=" * 60)
    
    try:
        from poker_ai.model.ppo_model import PPOModel
        from poker_ai.config.ppo_config import DEFAULT_PPO_CONFIG
        import tensorflow as tf
        
        model = PPOModel(config=DEFAULT_PPO_CONFIG)
        
        # 测试不同熵值下的自适应系数
        test_probs = [
            np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32),  # 均匀分布（高熵）
            np.array([0.8, 0.05, 0.05, 0.05, 0.05], dtype=np.float32),  # 集中分布（低熵）
        ]
        
        for i, probs in enumerate(test_probs):
            probs_tensor = tf.convert_to_tensor(probs.reshape(1, -1), dtype=tf.float32)
            mask = tf.convert_to_tensor([1.0], dtype=tf.float32)
            
            entropy_loss, adaptive_coef = model._compute_entropy_loss(probs_tensor, mask)
            
            logger.info(f"  测试{i+1}:")
            logger.info(f"    概率分布: {probs}")
            logger.info(f"    熵损失: {entropy_loss.numpy():.4f}")
            logger.info(f"    自适应系数: {adaptive_coef:.4f}")
        
        logger.info(f"✓ 自适应熵系数测试成功")
        return True
    except Exception as e:
        logger.error(f"✗ 自适应熵系数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """测试配置参数"""
    logger.info("=" * 60)
    logger.info("测试6: PPO配置参数")
    logger.info("=" * 60)
    
    try:
        from poker_ai.config.ppo_config import PPOConfig, DEFAULT_PPO_CONFIG
        
        # 测试默认配置
        logger.info("默认配置参数:")
        config_dict = DEFAULT_PPO_CONFIG.to_dict()
        for key, value in config_dict.items():
            logger.info(f"  {key}: {value}")
        
        # 测试自定义配置
        custom_config = PPOConfig(
            lr=5e-4,
            warmup_steps=2000,
            entropy_coef_target=0.015,
            value_clip=0.3
        )
        
        logger.info("自定义配置参数:")
        custom_dict = custom_config.to_dict()
        for key, value in custom_dict.items():
            logger.info(f"  {key}: {value}")
        
        logger.info(f"✓ 配置测试成功")
        return True
    except Exception as e:
        logger.error(f"✗ 配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    logger.info("PPO优化验证测试开始")
    logger.info(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    tests = [
        ("网络架构", test_network_architecture),
        ("学习率调度", test_lr_scheduler),
        ("GAE计算", test_gae_computation),
        ("训练步骤", test_training_step),
        ("自适应熵系数", test_entropy_adaptive),
        ("配置参数", test_config),
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
        logger.info("")
    
    # 输出测试结果
    logger.info("=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)
    
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{test_name}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    
    logger.info("=" * 60)
    logger.info(f"总计: {passed}/{total} 个测试通过")
    logger.info("=" * 60)
    
    if passed == total:
        logger.info("🎉 所有测试通过！PPO优化正常工作！")
        return 0
    else:
        logger.error("❌ 部分测试失败，请检查优化实现")
        return 1

if __name__ == "__main__":
    sys.exit(main())
