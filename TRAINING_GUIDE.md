# 德州扑克AI模型训练指南

## 训练流程详解

本指南提供了训练德州扑克AI模型的详细步骤、参数配置和最佳实践。通过本指南，您可以系统地训练、评估和优化您的AI模型。

## 训练前准备

### 环境配置

确保您的环境满足以下要求：

- Python 3.8+
- TensorFlow 2.5+
- 足够的内存（推荐16GB+）
- GPU支持（强烈推荐，可显著加速训练）

### 依赖安装

```bash
pip install -r requirements.txt
```

## 基本训练命令

### PPO 训练（推荐）

```bash
python train.py --mode train --model-type ppo --num-hands 100000 --learning-rate 3e-4 --ppo-clip 0.2 --ppo-epochs 10
```

PPO 使用轨迹级更新：每手牌收集 (state, action, old_prob)，牌局结束时用最终收益计算 GAE 并更新策略。检查点保存为 `base.actor.weights.h5`、`base.critic.weights.h5` 与 `base_state.pkl`，加载时指定 base 路径（如 `--model-path ./checkpoints/agent_ppo_1000`）。

### DQN 训练

```bash
python train.py --mode train --model-type dqn --num-hands 100000 --learning-rate 0.001 --batch-size 64
```

### 从检查点继续训练

```bash
python train.py --mode train --resume --checkpoint-dir ./checkpoints
```

## 训练参数详细说明

### 核心训练参数

| 参数 | 描述 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--mode` | 运行模式 | train | train/eval/demo |
| `--model-type` | 模型类型 | ppo | ppo/dqn/policy_gradient |
| `--num-hands` | 训练的手牌数量 | 100000 | 100000-1000000 |
| `--learning-rate` | 学习率 | 0.001 | 0.0001-0.001 |
| `--batch-size` | 批处理大小 | 64 | 32-128 |
| `--eval-interval` | 评估间隔（手牌数） | 1000 | 500-2000 |
| `--checkpoint-interval` | 保存检查点间隔（手牌数） | 5000 | 5000-20000 |

### 高级参数

| 参数 | 描述 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--discount-factor` | 奖励折扣因子 | 0.95 | 0.9-0.99 |
| `--eps-start` | ε-贪婪策略初始探索率 | 1.0 | 0.8-1.0 |
| `--eps-end` | ε-贪婪策略最终探索率 | 0.01 | 0.01-0.1 |
| `--eps-decay` | ε衰减速率 | 0.0001 | 0.00001-0.001 |
| `--target-update` | 目标网络更新频率 | 100 | 50-200 |
| `--replay-buffer-size` | 经验回放缓冲区大小 | 10000 | 5000-100000 |
| `--hidden-units` | 隐藏层神经元数量 | [256, 128] | [128, 64]-[512, 256] |
| `--optimizer` | 优化器类型 | adam | adam/adamax/rms_prop |

### 自我对弈参数

| 参数 | 描述 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--use-self-play` | 启用自我对弈训练 | True | True/False |
| `--opponent-update-interval` | 对手更新间隔 | 10000 | 5000-20000 |
| `--opponent-horizon` | 对手历史版本数量 | 5 | 3-10 |
| `--self-play-mixture` | 对手混合比例 | [0.5, 0.3, 0.2] | 调整以匹配对手历史版本数量 |

## 训练阶段建议

### 1. 初始训练阶段（10万-20万手牌）

在这个阶段，模型主要学习基本的德州扑克概念和规则。建议使用以下配置：

```bash
python train.py --mode train --model-type dqn --num-hands 200000 --learning-rate 0.001 --batch-size 64 --eps-start 1.0 --eps-end 0.1
```

这个阶段的关键是让模型广泛探索各种可能的动作，建立对手牌强度和基本策略的理解。

### 2. 优化训练阶段（30万-50万手牌）

在这个阶段，逐渐降低学习率，增加训练稳定性：

```bash
python train.py --mode train --resume --num-hands 500000 --learning-rate 0.0005 --batch-size 64 --target-update 200
```

### 3. 精细调优阶段（50万-100万手牌）

在最后阶段，使用更低的学习率进行精细调整，并更频繁地评估模型：

```bash
python train.py --mode train --resume --num-hands 1000000 --learning-rate 0.0001 --eval-interval 500 --save-best-only
```

## 自我迭代训练策略

### 对手更新策略

系统支持多种对手更新策略：

1. **固定间隔更新**：每训练N局后，将主代理的版本复制给对手
2. **性能触发更新**：当主代理的性能超过当前对手一定阈值时更新对手
3. **混合对手池**：维护多个历史版本的对手，随机选择进行对弈

可以通过以下参数调整：

```bash
python train.py --mode train --use-self-play --opponent-update-interval 10000 --opponent-horizon 5
```

### 渐进式难度设置

对于更有效的训练，可以从较简单的对手开始，逐步增加难度：

1. 先与随机对手进行训练（`--initial-opponent random`）
2. 然后与固定策略对手训练（`--initial-opponent fixed`）
3. 最后启用完全自我对弈（`--use-self-play`）

## 训练监控

### 日志监控

训练过程中，系统会输出详细的日志信息。关键指标包括：

- 每手牌的平均奖励
- 当前胜率
- 平均底池大小
- 模型损失
- ε值（探索率）

### 定期评估

使用`evaluate.py`脚本定期评估模型性能：

```bash
python evaluate.py --model-path ./checkpoints/best_agent_dqn.h5 --num-games 100 --opponent-type previous --generate-plots
```
ssh -p 12272 root@connect.cqa1.seetacloud.com 
评估报告将生成在`./eval_reports`目录下，包含胜率、收益分布、动作选择分布等详细信息。

## 模型优化技巧

### 1. 超参数调优

训练效果不佳时，可尝试调整以下关键超参数：

- **学习率**：如果损失波动大，尝试降低学习率；如果学习停滞，尝试适当提高
- **批处理大小**：增加批处理大小可提高训练稳定性，但会增加内存消耗
- **折扣因子**：增加折扣因子可以使模型更关注长期收益
- **探索参数**：调整ε-贪婪策略的参数，平衡探索和利用

### 2. 奖励函数调整

奖励函数对训练效果有重大影响。默认的奖励函数考虑了多个因素，但您可以根据需要进行调整。

修改`reward/reward_calculator.py`中的奖励计算逻辑：

```python
# 示例：增加位置优势权重
def calculate_reward(game_state, action, result):
    # 基本奖励（直接收益）
    base_reward = result['profit']
    
    # 位置优势奖励
    position_bonus = 0.5 * (game_state['players'] - game_state['current_player_pos']) / game_state['players']
    
    # 手牌强度奖励
    hand_strength_bonus = 0.3 * result['hand_strength']
    
    # 最终奖励
    total_reward = base_reward + position_bonus + hand_strength_bonus
    
    return total_reward
```

### 3. 网络结构优化

根据需要调整神经网络结构，以适应不同的复杂度要求：

```bash
python train.py --mode train --hidden-units 512 256 128
```

## 常见训练问题及解决方案

### 1. 训练不稳定

**症状**：奖励波动剧烈，模型性能不稳定

**解决方案**：
- 降低学习率
- 增加批处理大小
- 使用梯度裁剪
- 减少模型复杂度
- 增加经验回放缓冲区大小

### 2. 学习停滞

**症状**：奖励长时间不再提高

**解决方案**：
- 调整ε参数，增加探索
- 尝试不同的优化器
- 增加模型复杂度
- 调整奖励函数
- 启用学习率衰减

### 3. 过拟合

**症状**：训练性能良好，但评估性能差

**解决方案**：
- 添加正则化
- 使用dropout
- 增加数据多样性
- 提前停止训练

### 4. 训练速度慢

**症状**：每手牌训练时间过长

**解决方案**：
- 减少模型复杂度
- 调整批量大小
- 确保使用GPU加速
- 减少评估频率
- 使用更高效的手牌评估算法

## 资源管理

### 存储管理

训练过程中会生成大量检查点文件，定期清理不再需要的文件：

```bash
# 只保留最近10个检查点
sort -r ./checkpoints/model_*.h5 | tail -n +11 | xargs rm
```

### 训练时间估计

- 10万手牌：1-3天（取决于硬件）
- 50万手牌：1-2周
- 100万手牌：2-4周

## 高级训练技术

### 1. 集成学习

训练多个不同初始化的模型，然后集成它们的预测结果：

```bash
python train.py --mode train --use-ensemble --ensemble-size 3 --ensemble-voting soft
```

### 2. 模仿学习初始化

使用专家玩家的数据进行预训练，然后再使用强化学习优化：

```bash
python train.py --mode imitation --expert-data ./expert_data/hands.json --imitation-epochs 50
python train.py --mode train --resume --imitation-init
```

### 3. 分层训练

分别针对不同的游戏阶段（翻牌前、翻牌、转牌、河牌）进行专门训练：

```bash
python train.py --mode train --stage preflop --num-hands 100000
python train.py --mode train --stage flop --num-hands 100000
python train.py --mode train --stage turn --num-hands 100000
python train.py --mode train --stage river --num-hands 100000
python train.py --mode train --stage all --num-hands 500000
```

## 最佳实践总结

1. **从简单到复杂**：先训练基本策略，再添加复杂决策
2. **监控训练进度**：定期检查日志和评估报告
3. **使用检查点**：频繁保存模型，以便在训练中断后恢复
4. **多样化训练**：使用不同的初始参数训练多个模型
5. **逐步调优**：先调整大参数，再微调小参数
6. **验证假设**：每次只改变一个参数，观察效果
7. **耐心训练**：德州扑克是复杂的游戏，需要大量训练数据才能达到好的效果

## 训练输出说明

训练完成后，您将获得以下文件：

- 模型文件：`.h5`或`.pb`格式的神经网络模型
- 检查点文件：保存训练状态的文件
- 训练日志：记录训练过程和性能指标
- 评估报告：模型性能的详细分析

## 后续步骤

训练完成后，您可以：

1. 使用`evaluate.py`进行更详细的评估
2. 使用`--mode demo`参数演示模型表现
3. 在自己的德州扑克应用中集成AI模型
4. 基于训练的模型实现AI辅助建议功能

祝您训练顺利！如有任何问题，请参考README.md中的常见问题部分或联系项目维护者。