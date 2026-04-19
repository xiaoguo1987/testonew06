# SEED RL 训练架构 — 代码模块与类说明

## 架构总览

SEED RL 采用 **GPU-Centric 集中式推理** 架构：
- **主进程（GPU）**：维护 Actor/Critic/Opponent 三个神经网络，执行所有推理和 PPO 训练
- **Worker 进程（CPU×64）**：每个管理 8 张牌桌，只跑纯游戏逻辑，不加载 TensorFlow

```
train.py (入口)
├── poker_ai/training/gpu_trainer.py      ← SeedRLTrainer（GPU主训练器）
│   ├── poker_ai/model/large_ppo_model.py ← LargePPOModel（14M参数PPO模型）
│   │   └── poker_ai/config/ppo_config.py ← PPOConfig（训练超参数）
│   └── poker_ai/training/env_worker.py   ← env_worker_loop（CPU游戏Worker）
│       ├── poker_ai/game/poker_game.py   ← PokerGame, Player（游戏核心逻辑）
│       ├── poker_ai/game/cards.py        ← Card, Deck（牌组定义）
│       ├── poker_ai/model/state_representation.py ← StateRepresentation（134维状态编码）
│       ├── poker_ai/model/hand_evaluator.py       ← HandEvaluator（手牌评估）
│       └── poker_ai/reward/reward_calculator.py   ← RewardCalculator（奖励计算）
```

---

## 1. `train.py` — 训练入口

| 函数 | 作用 |
|------|------|
| `parse_arguments()` | 解析命令行参数（`--arch seed-rl`, `--hardware-profile rtx4090d` 等） |
| `_apply_hardware_profile(args)` | 根据硬件档位设置优化参数：隐藏层 [2048,2048,1024,512]、batch=2048、lr=1e-4、entropy=0.05、禁用 FP16 |
| `setup_environment(args)` | 配置 TensorFlow GPU 显存、XLA 编译、混合精度开关 |
| `_train_seed_rl(args)` | SEED RL 训练入口：创建 PPOConfig → 构建 LargePPOModel → 初始化 SeedRLTrainer → 加载检查点 → 启动训练 |

---

## 2. `poker_ai/training/gpu_trainer.py` — GPU 主训练器

### 类：`SeedRLTrainer`

SEED RL 的核心调度中心，负责 Worker 管理、GPU 推理调度、PPO 训练触发。

| 方法 | 作用 |
|------|------|
| `__init__(model, num_workers, ...)` | 初始化模型、创建独立对手模型 `opp_model`、设置推理批量参数 |
| `train(num_hands)` | **主训练循环**：启动 Worker 进程 → 循环执行（推理调度 → 轨迹收集 → PPO 训练 → 评估/保存） |
| `_dispatch_inference(pipes, pending)` | **GPU 推理调度引擎**：超时积攒式批量收集请求（min_batch=32, max_batch=512），区分 Main_AI 和 PPO 对手分别推理 |
| `_collect_trajectories(traj_queue, ...)` | 从 Queue 非阻塞读取完成手牌，更新 traj_buffer |
| `_do_ppo_train(traj_buffer)` | buffer ≥ 4096 步时触发 PPO 训练，调用 `model.train()` |
| `_run_evaluation(num_games)` | 串行评估：当前模型 vs 随机对手，返回胜率/平均收益 |
| `save_checkpoint()` / `load_checkpoint()` | 检查点管理 |

**关键设计**：
- **独立对手模型**：`self.opp_model` 独立实例，消除频繁 `set_weights` 开销
- **背压机制**：buffer 超阈值时暂停推理、优先训练，防止 OOM
- **小 buffer 保护**：训练结束时 buffer < max(256, batch_size) 则丢弃，防止过拟合

---

## 3. `poker_ai/training/env_worker.py` — CPU 游戏 Worker

### 函数：`env_worker_loop(worker_id, state_pipe, traj_queue, game_cfg, opp_type_list, total_hands)`

每个 Worker 进程的主循环，管理 `tables_per_worker=8` 张并发牌桌。**不导入 TensorFlow**。

| 内部函数 | 作用 |
|----------|------|
| `_advance_table_until_infer(table)` | 推进牌桌直到需要 GPU 推理（遇到 Main_AI 或 PPO 对手时返回推理请求） |
| `_apply_infer_result(table, ...)` | 接收 GPU 推理结果，执行动作采样并记录轨迹 |
| `_apply_mask_and_sample(action_probs, game, player, sr)` | 合法动作掩码 → 翻牌前位置过滤 → 概率采样 |
| `_settle_table(table)` | 手牌结算：计算奖励（含 Anti-Fold-Bot 塑形 + All-In 补偿）→ 通过 Queue 回传轨迹 |

**通信协议**：
- `state_pipe.send(('infer', worker_id, player_name, state_134d))` → 发送推理请求
- `state_pipe.recv()` → 接收 `(action_probs_5d, value_scalar)`
- `traj_queue.put(('traj', result_dict, trajectory_dict))` → 回传轨迹

---

## 4. `poker_ai/model/large_ppo_model.py` — PPO 大模型

### 类：`LargePPOModel`

14.2M 参数的 Actor-Critic 网络（134→2048→2048→1024→512→5）。

| 方法 | 作用 |
|------|------|
| `__init__(config)` | 构建 Actor（softmax 输出）和 Critic（线性输出）网络，残差连接 + LayerNorm |
| `_build_network(softmax_output)` | 参数化网络构建，支持残差连接层配置 |
| `predict_batch(states)` | **@tf.function 加速的 GPU 批量推理**：输入 (N,134) → 输出 (N,5) 概率 + (N,) 价值 |
| `train(states, actions, rewards, dones, old_probs)` | **PPO 训练主函数**：GAE 计算 → Mini-batch 切分 → 多 Epoch 更新 → KL 早停 |
| `_ppo_update_step(...)` | **@tf.function 单次 PPO 更新**：Actor loss (PPO clip + 熵正则) + Critic loss (Huber + 值裁剪) |
| `_compute_gae(rewards, values, dones)` | GAE(λ=0.95) 优势函数计算，advantage clip [-5, 5] |
| `warmup()` | 模型预热，触发 TF 图编译消除首次推理延迟 |
| `get_weights()` / `set_weights()` | 权重导出/导入（用于对手模型同步） |

**数值稳定性保护**：
- 概率 clamp [1e-6, 1.0] + 重新归一化
- log_ratio clamp [-20, 20]
- KL clamp [0, 100]
- 梯度裁剪 max_grad_norm=0.5

---

## 5. `poker_ai/config/ppo_config.py` — 训练配置

### 类：`PPOConfig`（dataclass）

| 参数组 | 关键参数 |
|--------|----------|
| 网络结构 | `input_dim=134`, `output_dim=5`, `hidden_dims=[512,512,256,128]`（被 seed-rl 覆盖为 [2048,2048,1024,512]） |
| PPO 核心 | `clip_epsilon=0.2`, `gamma=0.99`, `gae_lambda=0.95`, `ppo_epochs=10` |
| 正则化 | `entropy_coef=0.02`（seed-rl 覆盖为 0.05）, `kl_target=0.15`, `kl_early_stop=True` |
| 优化器 | `lr=3e-4`（seed-rl 覆盖为 1e-4）, `max_grad_norm=0.5`, `value_clip=0.2` |
| Buffer | `ppo_buffer_size=4096`, `batch_size=2048` |

---

## 6. `poker_ai/game/poker_game.py` — 游戏核心

### 类：`Player`

| 属性/方法 | 作用 |
|-----------|------|
| `name, stack, bet, cards, is_active, is_all_in` | 玩家状态 |
| `add_bet(amount)` | 下注扣减筹码 |
| `reset_for_new_round()` | 新一轮重置 |

### 类：`PokerGame`

| 方法 | 作用 |
|------|------|
| `start_new_hand()` | 初始化手牌：洗牌、发牌、收盲注 |
| `handle_action(player, action, amount)` | 执行玩家动作（fold/call/raise/allin） |
| `next_player()` | 轮转到下一个活跃玩家 |
| `is_round_over()` / `is_hand_over()` | 检查轮次/手牌是否结束 |
| `proceed_to_next_round()` | 进入下一轮（翻牌/转牌/河牌） |

---

## 7. `poker_ai/game/cards.py` — 牌组定义

| 类 | 作用 |
|----|------|
| `Card` | 扑克牌：rank(2-A) + suit(♠♥♦♣)，支持整数编码 `card_code` |
| `Deck` | 52 张标准牌组：shuffle / deal |

---

## 8. `poker_ai/model/state_representation.py` — 状态编码

### 类：`StateRepresentation`

将游戏状态编码为 **134 维向量**：

| 维度 | 内容 |
|------|------|
| 0-51 | 玩家手牌 one-hot (52d) |
| 52-103 | 公共牌 one-hot (52d) |
| 104-121 | 所有玩家信息 (6人 × 3特征: stack/bet/active = 18d) |
| 122-126 | 游戏全局 (round/pot/current_bet/bb/sb = 5d) |
| 127-132 | 英雄位置 one-hot (6d: BTN/SB/BB/UTG/MP/CO) |
| 133 | 活跃玩家数 (1d) |

| 关键方法 | 作用 |
|----------|------|
| `get_state(game, player)` | 获取 134 维状态向量 |
| `calculate_hand_strength(player, game)` | 翻牌前 O(1) 查表 / 翻牌后蒙特卡洛模拟 |
| `decode_action(action_vector, game, player, hs)` | 5 维概率 → 具体动作 + 下注金额 |

---

## 9. `poker_ai/model/hand_evaluator.py` — 手牌评估

### 类：`HandEvaluator`

| 方法 | 作用 |
|------|------|
| `evaluate(cards)` | 评估 5-7 张牌的最佳牌型和等级 |
| `compare_hands(hand1, hand2)` | 比较两手牌强弱 |

---

## 10. `poker_ai/reward/reward_calculator.py` — 奖励计算

### 类：`RewardCalculator`

11 维精细化奖励系统：

| 奖励维度 | 权重 | 说明 |
|----------|------|------|
| `winning_hand` | 1.0 | 赢牌核心奖励 |
| `losing_hand` | 0.5 | 输牌惩罚 |
| `hand_strength` | 0.4 | 手牌强度一致性 |
| `pot_odds` | 0.3 | 底池赔率决策 |
| `aggressive_play` | 0.15 | 主动进攻奖励 |
| `vpip_bonus` | 0.08 | 好手牌主动入池 |
| `fold_when_weak` | 0.3 | 弱牌及时弃牌 |
| `bluff_success` | 0.3 | 成功诈唬 |
| `invalid_bet` | 0.6 | 无效下注惩罚 |
| `position_play` | 0.2 | 位置优势 |
| `consistency` | 0.1 | 决策一致性 |

| 关键方法 | 作用 |
|----------|------|
| `calculate_reward(player, game, action, ...)` | 综合计算单步奖励 |
| `_preflop_lut_lookup(cards, num_opponents)` | O(1) 翻牌前胜率查表 |

---

## 数据流总览

```
Worker (CPU)                          Main (GPU)
─────────────                         ──────────
推进牌桌 → 需要推理                    
  ↓                                    
pipe.send(state_134d) ──────────────→ _dispatch_inference()
                                       积攒 32-512 个请求
                                       ↓
                                       model.predict_batch() [GPU]
                                       ↓
pipe.recv(probs_5d, value) ←──────── 分发结果
  ↓
采样动作 → 执行 → 手牌结束
  ↓
queue.put(trajectory) ──────────────→ _collect_trajectories()
                                       积累到 traj_buffer
                                       ↓ buffer ≥ 4096
                                       model.train() [PPO更新]
```
