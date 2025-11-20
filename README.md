# 德州扑克AI模型

## 项目概述

本项目是一个基于强化学习的德州扑克AI系统，专为6人桌cash游戏设计。系统具有以下核心功能：

- 完整的德州扑克游戏逻辑实现
- 基于强化学习的AI决策模型
- 自我迭代训练系统，通过奖励机制不断优化
- AI辅助建议功能，为玩家提供实时决策支持
- 全面的训练和评估工具

## 目录结构

```
poker_ai/                   # 主项目目录
├── game/                   # 游戏逻辑模块
│   ├── __init__.py
│   ├── poker_game.py       # 德州扑克游戏核心实现
│   ├── card.py             # 卡牌相关类
│   └── player.py           # 玩家相关类
├── model/                  # AI模型模块
│   ├── __init__.py
│   ├── ai_agent.py         # AI代理实现
│   ├── dqn_model.py        # DQN模型实现
│   ├── pg_model.py         # 策略梯度模型实现
│   └── hand_evaluator.py   # 手牌评估器
├── reward/                 # 奖励机制模块
│   ├── __init__.py
│   ├── reward_calculator.py # 奖励计算器
│   └── self_play_trainer.py # 自我对弈训练器
├── advisor/                # AI建议器模块
│   ├── __init__.py
│   └── ai_advisor.py       # AI辅助建议器实现
train.py                    # 训练脚本
evaluate.py                 # 评估脚本
requirements.txt            # 依赖列表
README.md                   # 项目文档
```

## 安装说明

### 环境要求

- Python 3.8+
- TensorFlow 2.5+
- NumPy
- Matplotlib
- Seaborn

### 安装步骤

1. 克隆或下载项目代码到本地

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 确保项目目录结构正确，所有模块都在适当的位置

## 训练模型

### 基本训练

使用`train.py`脚本进行模型训练：

```bash
python train.py --mode train --model-type dqn --num-hands 10000
```

### 训练参数说明

- `--mode`: 运行模式，可选'train'(训练)、'eval'(评估)、'demo'(演示)
- `--model-type`: 模型类型，可选'dqn'或'policy_gradient'
- `--num-hands`: 训练的手牌数量
- `--learning-rate`: 学习率（默认为0.001）
- `--batch-size`: 批处理大小（默认为64）
- `--eval-interval`: 评估间隔（默认为100局）
- `--checkpoint-interval`: 保存检查点间隔（默认为500局）
- `--checkpoint-dir`: 检查点保存目录（默认为'./checkpoints'）

### 继续训练

要从之前的检查点继续训练，可以使用`--resume`参数：

```bash
python train.py --mode train --resume
```

或者指定具体的模型文件：

```bash
python train.py --mode train --model-path ./checkpoints/best_agent_dqn_20231120_120000.h5
```

### 高级训练选项

- `--use-ensemble`: 使用集成模型
- `--save-best-only`: 仅保存表现最好的模型
- `--verbose`: 输出详细日志

## 评估模型

### 基本评估

使用`evaluate.py`脚本进行详细评估：

```bash
python evaluate.py --model-path ./checkpoints/best_agent_dqn_20231120_120000.h5 --num-games 100
```

### 评估参数说明

- `--model-path`: 要评估的模型路径（必需）
- `--model-type`: 模型类型
- `--num-games`: 评估的游戏局数
- `--opponent-type`: 对手类型，可选'random'(随机)、'fixed'(固定策略)、'previous'(历史版本)
- `--output-dir`: 评估报告输出目录
- `--generate-plots`: 生成可视化图表

## 使用AI辅助建议功能

### 在Python代码中使用

```python
from poker_ai.advisor.ai_advisor import AIAvisor
from poker_ai.game.poker_game import PokerGame, Player

# 创建AI建议器
advisor = AIAvisor(model_path='./checkpoints/best_agent_dqn.h5')

# 创建游戏实例和玩家
# ...（游戏初始化代码）

# 获取建议
advice = advisor.get_detailed_advice(game, player)
print(advice['reasoning'])  # 打印建议理由
print(advice['recommended_action'])  # 打印推荐动作
print(advice['expected_value'])  # 打印预期价值
```

### 使用演示模式

使用`train.py`的演示模式来查看AI的实际表现：

```bash
python train.py --mode demo --model-path ./checkpoints/best_agent_dqn.h5 --demo-hands 10
```

## 自我迭代训练原理

本系统使用以下机制实现自我迭代：

1. **自我对弈**：AI与自身的历史版本进行对弈，不断挑战自我
2. **自适应奖励机制**：根据游戏表现动态调整奖励函数
3. **定期更新对手**：周期性地将主代理的版本复制给对手，保持挑战性
4. **性能评估**：定期评估模型性能，并保存表现最佳的版本
5. **渐进式学习**：从简单策略开始，逐步过渡到复杂策略

### 奖励函数设计

奖励函数考虑了以下因素：
- 直接收益/损失
- 手牌强度
- 底池赔率
- 位置优势
- 动作的预期价值
- 游戏阶段（翻牌前、翻牌、转牌、河牌）

## 代码使用示例

### 创建游戏并运行

```python
from poker_ai.game.poker_game import PokerGame, Player
from poker_ai.model.ai_agent import PokerAIAgent

# 创建玩家
players = [
    Player("Player1", 1000, is_human=False),
    Player("Player2", 1000, is_human=False),
    Player("Player3", 1000, is_human=False),
    Player("Player4", 1000, is_human=False),
    Player("Player5", 1000, is_human=False),
    Player("Player6", 1000, is_human=False)
]

# 创建游戏
game = PokerGame(small_blind=5, big_blind=10, players=players)

# 运行游戏
for _ in range(10):  # 运行10手牌
    game.start_new_hand()
    while not game.is_round_over():
        # 这里添加AI决策逻辑
        pass
```

### 使用AI代理

```python
from poker_ai.model.ai_agent import PokerAIAgent

# 创建AI代理
agent = PokerAIAgent(agent_type='dqn')

# 加载模型
agent.load_model('./checkpoints/best_agent_dqn.h5')

# 选择动作
action, amount = agent.select_action(game, current_player)
```

## 训练技巧

1. **从小规模开始**：先使用较小的手牌数量（如1000-5000）进行训练，确保系统正常工作
2. **监控训练进度**：定期检查日志和评估报告，了解模型性能
3. **调整超参数**：如果训练效果不佳，尝试调整学习率、批处理大小等超参数
4. **使用检查点**：训练中断后可以从最近的检查点继续
5. **定期评估**：使用`evaluate.py`进行详细评估，生成可视化报告分析模型表现

## 常见问题

### 训练速度慢

- 减少每次评估的游戏局数（`--num-eval-games`）
- 增加检查点间隔（`--checkpoint-interval`）
- 如果有GPU，确保TensorFlow正确使用GPU

### 模型性能不佳

- 增加训练的手牌数量
- 调整学习率
- 尝试不同的模型类型（DQN vs 策略梯度）
- 检查奖励函数是否合理

### 内存错误

- 减少批处理大小
- 减少同时训练的游戏数量
- 清理不再需要的检查点文件

## 许可证

[MIT License](LICENSE)

## 贡献

欢迎提交问题报告和改进建议！

## 联系信息

如有任何问题，请联系项目维护者。