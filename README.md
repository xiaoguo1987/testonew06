# 德州扑克 AI（6-Max）

基于强化学习的 6 人桌德州扑克训练项目，包含：

- 自我对弈训练（`train.py`）
- 评估与报告（`train.py --mode eval`、`evaluate.py`）
- 演示模式（`train.py --mode demo`）
- AI 建议器（`poker_ai/advisor/ai_advisor.py`）

---

## 1. 快速开始

### 1.1 环境准备

- Python 3.8+
- TensorFlow 2.x（项目当前在 `2.9.0` 环境下可运行）

安装依赖：

```bash
pip install -r requirements.txt
```

### 1.2 先跑一个 500 手牌验证

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

训练日志会写到 `logs/`，检查点默认写到 `checkpoints/`（或你指定目录）。

---

## 2. 项目结构（当前代码对应）

```text
train.py                         # 主入口：train/eval/demo
evaluate.py                      # 详细评估脚本（可生成图表）
requirements.txt
TRAINING_GUIDE.md                # 历史训练说明（部分参数已过期）
docs/如何看训练结果.md

poker_ai/
  game/
    cards.py                     # 牌与牌型评估、compare_hands
    poker_game.py                # 6-max 游戏流程与结算
  model/
    ai_agent.py                  # 统一代理接口（dqn / policy_gradient / ppo）
    dqn_agent.py                 # DQN + PolicyGradient
    ppo_model.py                 # PPO Actor-Critic
    state_representation.py      # 状态编码（134 维）
    hand_evaluator.py
  reward/
    self_play_trainer.py         # 自我对弈训练主循环
    reward_calculator.py
    simple_reward_calculator.py
  advisor/
    ai_advisor.py                # 建议器（类名是 AIAvisor）
  config/
    ppo_config.py
  utils/
    checkpoint.py
```

---

## 3. 训练与评估命令

### 3.1 训练（推荐 PPO）

```bash
python train.py --mode train --model-type ppo --num-hands 10000
```

### 3.2 从最近检查点继续训练

```bash
python train.py --mode train --model-type ppo --resume
```

### 3.3 指定检查点继续训练

PPO 传 **base 路径**（不带后缀）：

```bash
python train.py --mode train --model-type ppo \
  --model-path ./checkpoints/agent_ppo_1000
```

DQN/PG 可传 `.weights.h5` 或 `.pkl` 路径。

### 3.4 用 `train.py` 做标准评估（推荐）

```bash
python train.py --mode eval --model-type ppo \
  --model-path ./checkpoints/agent_ppo_1000 \
  --num-eval-games 500 \
  --eval-output eval_results.json
```

### 3.5 用 `evaluate.py` 做详细评估（图表）

```bash
python evaluate.py \
  --model-path ./checkpoints/agent_dqn_1000.weights.h5 \
  --model-type dqn \
  --num-games 200 \
  --generate-plots
```

---

## 4. 模型与检查点说明

### 4.1 支持的模型类型

- `dqn`
- `policy_gradient`
- `ppo`

### 4.2 PPO 检查点文件

PPO 保存为三件套（同一个 base）：

- `*.actor.weights.h5`
- `*.critic.weights.h5`
- `*_state.pkl`

例如 base 为 `./checkpoints/agent_ppo_500`，对应：

- `./checkpoints/agent_ppo_500.actor.weights.h5`
- `./checkpoints/agent_ppo_500.critic.weights.h5`
- `./checkpoints/agent_ppo_500_state.pkl`

---

## 5. 常用参数（以 `train.py` 实际 argparse 为准）

### 基础参数

- `--mode {train,eval,demo}`
- `--model-type {dqn,policy_gradient,ppo}`
- `--num-hands`
- `--checkpoint-dir`
- `--resume`
- `--model-path`

### PPO 相关

- `--learning-rate`（默认 `3e-4`）
- `--ppo-clip`（默认 `0.2`）
- `--ppo-epochs`（默认 `10`）
- `--ppo-entropy-coef`（默认 `0.02`）

### 训练过程控制

- `--eval-interval`（默认 `200`）
- `--num-eval-games`（默认 `100`）
- `--checkpoint-interval`（默认 `1000`）
- `--skip-final-eval`
- `--target-update`（DQN 目标网络更新频率）

### 游戏配置

- `--small-blind`（默认 `5`）
- `--big-blind`（默认 `10`）
- `--initial-stack`（默认 `1000`）
- `--num-players`（默认 `6`）

---

## 6. 当前实现的关键约束（重要）

- 训练器是 **6-max 专用**：`SelfPlayTrainer` 要求 `num_players == 6`。
- `PokerGame` 也是 6 人桌实现。
- 项目核心依赖 TensorFlow，未安装会在导入阶段报错。
- 胜率统计当前按“本局是否为赢家之一”计算（支持并列赢家）。
- `evaluate.py` 的 `--model-type` 选项目前只允许 `dqn/policy_gradient`（脚本参数层面的限制），PPO 正式评估建议用 `train.py --mode eval`。

---

## 7. 如何判断训练是否正常

看日志中的三类信息：

- **运行状态**：是否有 `ERROR/Exception`
- **效率**：`平均每手耗时`（例如 0.15s/手）
- **效果**：`评估完成` 中的 `平均利润 / 胜率 / 诈唬率 / 加注率`

建议先做小样本（500~1000）验证流程，再上大手数。

---

## 8. 已知命名与兼容性备注

- 建议器类名拼写是 `AIAvisor`（不是 `AIAdvisor`）。
- 旧文档中的部分参数已过期，请以 `train.py`、`evaluate.py` 当前参数为准。

---

## 9. 一个最小 Python 调用示例

```python
from poker_ai.game.poker_game import PokerGame, Player
from poker_ai.model.ai_agent import PokerAIAgent

players = [Player(f"P{i+1}", 1000, True) for i in range(6)]
game = PokerGame(5, 10, players)
agent = PokerAIAgent(agent_type="ppo")

game.start_new_hand()
p = game.players[0]
action, amount = agent.select_action(game, p)
print(action, amount)
```