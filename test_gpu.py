#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速GPU测试脚本 - 验证GPU是否正常工作
"""

import time
import numpy as np

print("=" * 60)
print("GPU 快速测试")
print("=" * 60)

try:
    import tensorflow as tf
    
    # 检测GPU
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\n✓ 检测到 {len(gpus)} 个GPU")
    
    if len(gpus) == 0:
        print("✗ 没有检测到GPU！")
        exit(1)
    
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
    
    # 启用显存按需增长
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass
    
    # 测试矩阵乘法（会让GPU跑起来）
    print("\n开始GPU计算测试...")
    print("（在另一个终端运行 watch -n 1 nvidia-smi 观察GPU利用率）")
    
    with tf.device('/GPU:0'):
        # 创建大矩阵
        size = 10000
        print(f"创建 {size}x{size} 矩阵...")
        a = tf.random.normal([size, size])
        b = tf.random.normal([size, size])
        
        # 多次矩阵乘法（持续约10-20秒）
        print("执行矩阵乘法（持续15秒）...")
        print("现在你应该能看到GPU利用率上升！\n")
        
        start = time.time()
        for i in range(50):
            c = tf.matmul(a, b)
            if i % 10 == 0:
                elapsed = time.time() - start
                print(f"  迭代 {i}/50, 耗时: {elapsed:.2f}s")
        
        elapsed = time.time() - start
        print(f"\n✓ GPU计算完成！总耗时: {elapsed:.2f}s")
        print(f"  平均每次迭代: {elapsed/50*1000:.2f}ms")
    
    # 测试混合精度
    print("\n测试混合精度...")
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    
    with tf.device('/GPU:0'):
        a_fp16 = tf.cast(a, tf.float16)
        b_fp16 = tf.cast(b, tf.float16)
        
        start = time.time()
        for i in range(50):
            c = tf.matmul(a_fp16, b_fp16)
        elapsed = time.time() - start
        
        print(f"✓ 混合精度测试完成！总耗时: {elapsed:.2f}s")
        print(f"  平均每次迭代: {elapsed/50*1000:.2f}ms")
    
    print("\n" + "=" * 60)
    print("测试完成！GPU工作正常！")
    print("=" * 60)
    print("\n现在可以开始PPO训练了：")
    print("  python train.py --mode train --model-type ppo --num-hands 1000")
    
except ImportError as e:
    print(f"\n✗ 错误: {e}")
    print("请安装TensorFlow: pip install tensorflow")
except Exception as e:
    print(f"\n✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
