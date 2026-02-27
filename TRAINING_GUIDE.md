# 训练指南（当前版本）

本指南以当前仓库代码为准，目标是：

- 快速验证训练逻辑是否正常
- 用小样本（500~1000 手）观察指标是否合理
- 再逐步放大训练规模

---

## 1. 先做小样本验证

### 1.1 500 手验证（推荐先跑）

```bash
python train.py --mode train --model-type ppo \
  --num-hands 500 \
  --eval-interval 250 \
  --num-eval-games 100 \
  --checkpoint-interval 500 \
  --learning-rate 2e-4 \
  --ppo-clip 0.15 \
  --ppo-epochs 8 \
  --ppo-entropy-coef 0.006 \
  --target-update 20 \
  --checkpoint-dir ./checkpoints_run500_fixwinrate \
  --skip-final-eval
```

### 1.2 1000 手验证（第二步）

```bash
python train.py --mode train --model-type ppo \
  --num-hands 1000 \
  --eval-interval 250 \
  --num-eval-games 100 \
  --checkpoint-interval 500 \
  --learning-rate 2e-4 \
  --ppo-clip 0.15 \
  --ppo-epochs 8 \
  --ppo-entropy-coef 0.006 \
  --target-update 20 \
  --checkpoint-dir ./checkpoints_run1000_fixwinrate \
  --skip-final-eval
```

---

## 2. 训练参数说明（仅列当前代码有效参数）

来自 `train.py` 的 argparse：

### 基础

- `--mode {train,eval,demo}`
- `--model-type {dqn,policy_gradient,ppo}`
- `--model-path`
- `--checkpoint-dir`
- `--resume`
- `--num-hands`

### PPO

- `--learning-rate`（默认 `3e-4`）
- `--ppo-clip`（默认 `0.2`）
- `--ppo-epochs`（默认 `10`）
- `--ppo-entropy-coef`（默认 `0.02`）

### 训练流程

- `--eval-interval`（默认 `200`）
- `--num-eval-games`（默认 `100`）
- `--checkpoint-interval`（默认 `1000`）
- `--skip-final-eval`
- `--target-update`（主要用于 DQN）

### 游戏配置

- `--small-blind`（默认 `5`）
- `--big-blind`（默认 `10`）
- `--initial-stack`（默认 `1000`）
- `--num-players`（默认 `6`）

---

## 3. 检查点与恢复训练

### 3.1 PPO 文件格式

PPO 使用 base 路径，保存三件套：

- `base.actor.weights.h5`
- `base.critic.weights.h5`
- `base_state.pkl`

例如：`base = ./checkpoints/agent_ppo_500`

### 3.2 继续训练

自动找最新：

```bash
python train.py --mode train --model-type ppo --resume
```

指定模型：

```bash
python train.py --mode train --model-type ppo \
  --model-path ./checkpoints/agent_ppo_500
```

---

## 4. 评估方式

### 4.1 推荐：`train.py --mode eval`

```bash
python train.py --mode eval --model-type ppo \
  --model-path ./checkpoints/agent_ppo_1000 \
  --num-eval-games 500 \
  --eval-output eval_results.json
```

### 4.2 详细评估：`evaluate.py`

`evaluate.py` 适合出图表和报告，但当前参数限制 `--model-type` 只接受 `dqn/policy_gradient`。

---

## 5. 看日志时重点关注什么

每次训练建议先看这 4 类指标：

1. 是否有报错（`ERROR/Exception`）
2. 每手耗时（`平均每手耗时`）
3. 评估平均利润（`avg_profit`）
4. 评估胜率与行为指标（`win_rate / bluff_rate / raise_rate / invalid_bet_rate`）

如果 500~1000 手里 `avg_profit` 长期负且 `win_rate` 近 0，不建议直接加到超大手数，先调参数或奖励设计。

---

## 6. 当前版本已知约束

- 训练器是 6-max 专用（`SelfPlayTrainer` 要求 `num_players == 6`）。
- 游戏引擎是 6 人桌实现。
- 胜率按“本局是否为赢家之一”统计，支持并列赢家。
- 核心依赖 TensorFlow，未安装无法启动训练。

---

## 7. 常见问题

### Q1: 训练速度偏慢怎么办？

- 提高 `--eval-interval`（减少评估频率）
- 降低 `--num-eval-games`
- 使用 GPU 并确认 TensorFlow 识别到 GPU

### Q2: 为什么日志里训练阶段诈唬率常是 0？

训练阶段做了轻量统计以提速；评估阶段指标更有参考价值，优先看评估输出。

### Q3: 什么时候开始大规模训练？

先满足以下条件再放大：

- 无运行错误
- 500~1000 手评估指标不再明显异常
- 参数组合在两次以上小样本里表现一致

---

## 8. 推荐训练节奏

1. 500 手快速检查流程
2. 1000 手检查稳定性
3. 5000~10000 手开始做阶段性对比
4. 再进入更大规模训练