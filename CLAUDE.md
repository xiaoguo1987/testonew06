# CLAUDE.md — 扑克 AI 训练项目

## 项目概述

训练一个 **6-max 100bb NLHE（无限注德州扑克）AI 模型**，用于扑克辅助（advisor）系统。

- **项目路径**: `/root/testo-seed-rl`（服务器）/ `/Users/xiaoguo/Desktop/woaspaces/testo`（本地）
- **当前分支**: `codex/two-server-seed-rl-20260603`
- **训练架构**: SEED RL（GPU-Centric）+ PPO + 多对手集成
- **状态**: 研究进行中，尚未达到可交付标准

## 服务器信息

| 项目 | 详情 |
|------|------|
| SSH | `ssh -p 21477 root@connect.bjb1.seetacloud.com`（密钥已配置） |
| GPU | 2× RTX 4090 (24GB)，当前仅此一台在使用 |
| CPU | 128 vCPU |
| RAM | ~1 TiB |
| OS | Ubuntu 22.04 |
| Python | `/root/miniconda3/bin/python` (3.12.3) |
| TF | 2.21.0，两卡均可见 |
| 磁盘 | 仅 30GB，需关注空间！runs 目录占约 9GB |

## 可交付标准（agent_training_log.md 定义）

模型被认为可用的条件（当前均未满足）：
- [ ] 可复现的检查点保存为选定模型工件
- [ ] 截断率 < 1%
- [ ] 无效下注率 < 1%
- [ ] 动作分布不被单一捷径动作主导
- [ ] 最终报告说明已知局限性
- [ ] 重复验收门通过：avg >= 0, avg_lcb95 >= 0, worst >= -200, max row std <= 200

## 当前研究状态

### 核心问题：多目标优化冲突
对对手 A 的 PPO 提升会导致对对手 B 的退化。经过大量实验（加权采样、Anchor-KL、参考蒸馏、多教师蒸馏），目前没有任何检查点通过完整的重复验收。

### 已尝试且失败的方案
- 加权 PPO 修复（hard-negative / balanced / mild / focused / scored-only）
- 单对手旋转 / 集成训练 / 固定名册训练
- Anchor-KL 锚定修复 / 参考缓冲区策略蒸馏
- 多教师离线蒸馏（均值选择 → 已改为 LCB 选择）

### 最佳研究锚点
- `seed_rl_hands_3600064`：目前最不坏的基座（全池 avg -55.7，仍有多个弱行）
- `seed_rl_hands_3610064`：参考蒸馏候选，硬子集通过但全池失败
- `seed_rl_hands_3803147`：单次矩阵最佳但重复验收失败（假阳性）

### 历史对手池（13 个）
`1000054, 1000262, 1500036, 500016, 550046, 600040, 600047, 600049, 700016, 700169, 800200, 900022, 900236`

## 关键代码文件

### 核心训练
| 文件 | 作用 |
|------|------|
| `train.py` | 训练入口，参数解析，硬件配置 |
| `poker_ai/training/gpu_trainer.py` | SeedRLTrainer — GPU 主训练器（1811行） |
| `poker_ai/training/env_worker.py` | CPU 游戏 Worker — 多进程牌桌模拟 |
| `poker_ai/model/large_ppo_model.py` | LargePPOModel — 14M 参数 PPO 网络 |
| `poker_ai/model/ppo_model.py` | 原 PPO 模型（支持参考蒸馏） |
| `poker_ai/config/ppo_config.py` | PPOConfig dataclass |

### 评估与安全
| 文件 | 作用 |
|------|------|
| `scripts/evaluate_seed_pool.py` | 单检查点 vs 对手池评估 |
| `scripts/evaluate_seed_repeats.py` | 多种子重复评估（更稳定） |
| `scripts/evaluate_seed_repeats_parallel.py` | 多 GPU 并行重复评估 |
| `scripts/seed_matrix_gate.py` | 矩阵门控 PASS/FAIL 判断 |
| `scripts/seed_safety_validate.py` | 检查点安全验证（训练中回调） |
| `scripts/evaluate_seed_candidates.py` | 候选验收工作流 |
| `scripts/seed_run_report.py` | 训练运行报告 |

### 蒸馏与教师选择
| 文件 | 作用 |
|------|------|
| `scripts/select_seed_teachers.py` | 教师选择（支持 LCB 聚合） |
| `scripts/build_seed_reference_buffer.py` | 构建参考经验缓冲区 |
| `scripts/build_seed_multiteacher_buffer.py` | 多教师经验聚合 |
| `scripts/distill_seed_policy.py` | 离线策略蒸馏训练 |
| `scripts/check_seed_reference_kl.py` | 参考 KL 独立检查 |

### 游戏引擎
| 文件 | 作用 |
|------|------|
| `poker_ai/game/poker_game.py` | PokerGame, Player — 游戏核心逻辑 |
| `poker_ai/game/cards.py` | Card, Deck — 牌组定义 |
| `poker_ai/model/state_representation.py` | 134 维状态编码 |
| `poker_ai/model/hand_evaluator.py` | 手牌评估（翻牌前 LUT / 翻牌后 MC） |
| `poker_ai/reward/reward_calculator.py` | 11 维精细化奖励计算 |

### 配置文件
| 文件 | 作用 |
|------|------|
| `configs/seed_roster_3803147.json` | 固定对手名册 |
| `configs/seed_roster_3600064_weighted_safety.json` | 加权安全名册 |

## 重要注意事项

### 训练参数陷阱
- **硬件配置覆盖**: `rtx4090d` profile 会强制修改 `ppo_epochs=4`、`ppo_buffer_size=16384`，需显式传参覆盖
- **修复后**: 现在 `--ppo-epochs`、`--ppo-buffer-size` 等显式参数会被尊重
- **微更新建议**: `ppo_epochs=1`, `ppo_buffer_size=4096~8192`, `lr=1e-7~3e-8`

### 评估陷阱
- **100 局评估不可靠**: 高方差，会产生假阳性/假阴性
- **至少 500 局 × 3 重复**: 才能用于决策
- **均值假阳性**: `3803147` 单次矩阵 avg +103.9，重复验收 avg -87.3
- **LCB 门控必须**: 用 `--aggregate-repeats --rank-metric lcb` 选教师

### 磁盘空间
- 服务器只有 30GB 总空间，当前剩余约 20GB
- 清理旧运行时注意保留关键检查点
- 3M self-play 运行占 1.7GB（最大的单次运行）

### 不要做的事
- ❌ 不要从已被拒绝的检查点做 PPO 修复
- ❌ 不要用 20/100 局结果作为模型质量信号
- ❌ 不要在没有 1-GPU 独立验收的情况下提升检查点
- ❌ 不要把服务器密码或 API key 存仓库里
- ❌ 不要优化固定的 all-in 频率
- ❌ 不要仅靠自对弈 profit 判断模型质量

## 常用命令

### 训练
```bash
# 微更新 PPO 修复（推荐参数）
python train.py --arch seed-rl \
  --resume /path/to/checkpoint \
  --num-hands 50000 \
  --lr 1e-7 \
  --ppo-epochs 1 \
  --ppo-buffer-size 8192 \
  --ppo-clip 0.04 \
  --entropy-coef 0.01 \
  --anchor-policy-path /path/to/anchor \
  --anchor-kl-coef 0.1 \
  --reference-distill-coef 0.2 \
  --reference-file /path/to/reference.npz \
  --checkpoint-validation-command "python scripts/seed_safety_validate.py ..." \
  --checkpoint-interval 5000
```

### 评估
```bash
# 候选验收（完整流程）
python scripts/evaluate_seed_candidates.py \
  --candidate /path/to/checkpoint \
  --opponent-file /path/to/matrix.jsonl \
  --games 500 --repeats 3 \
  --out-dir /root/testo-seed-rl/runs/eval_XXX

# 安全验证（训练中回调）
python scripts/seed_safety_validate.py \
  --opponent-file /path/to/baseline_matrix.jsonl \
  --games 500 --repeats 3 \
  --min-avg 0 --min-worst -300 \
  --reference-file /path/to/reference.npz \
  --max-reference-mean-kl 0.01
```

### 教师选择与蒸馏
```bash
# LCB 教师选择
python scripts/select_seed_teachers.py \
  --matrix-glob "runs/repeat_*/raw/*.jsonl" \
  --aggregate-repeats --rank-metric lcb \
  --min-repeats 3 --max-std 250 \
  --out /tmp/teachers.json

# 多教师缓冲构建
python scripts/build_seed_multiteacher_buffer.py \
  --selection-file /tmp/teachers.json \
  --states-per-row 5000 \
  --out /tmp/multiteacher_buffer.npz

# 离线蒸馏
python scripts/distill_seed_policy.py \
  --student /path/to/base_checkpoint \
  --buffer /tmp/multiteacher_buffer.npz \
  --reference-file /path/to/reference.npz \
  --lr 3e-6 --epochs 5 \
  --out /path/to/distilled_checkpoint
```

## 2026-06-04 最新实验进展

### 🎉 140 维状态特征扩展 — 突破性进展！
**状态**: 134→140 维，新增 6 维对手身份特征（每座位：0=鱼, 1=PPO对手）

**修改的文件**:
- `poker_ai/model/state_representation.py` — 新增 `opponent_type_dim=6`, `get_opponent_type_vector()`
- `poker_ai/training/env_worker.py` — 传递 `opp_type_list` 到状态编码
- `poker_ai/training/gpu_trainer.py` — 评估时传递对手类型
- `poker_ai/config/ppo_config.py` — `input_dim=140`
- `poker_ai/model/large_ppo_model.py` — 自动权重扩展（134→140）
- `poker_ai/engine/poker_engine.cpp` — C++ 引擎扩展至 140 维

**关键设计**: 旧检查点（134维）自动扩展—前 134 行保留原权重，后 6 行初始化为零

### 首个通过安全验证的检查点！
- **检查点**: `3615185`（140 维，3 次 PPO 更新）
- **结果**: avg **+67.2**, worst -296.2 (600040)
- **对比 134 维最佳**: avg -71.0 → **+67.2**（139 点改善！）
- **C++ 引擎**: 已更新至 140 维，~320 hands/s

### 关键教训（新增）
- `--model-path` 指定检查点路径，`--resume` 是布尔标志
- 必须显式设置 `--train-workers 16 --tables-per-worker 48`
- `--hardware-profile rtx4090d` 会覆盖 ppo_epochs/ppo_buffer_size，需显式传参
- **C++ 引擎需重新编译**（`python setup_engine.py build_ext --inplace`）
- **记得清理 .pyc 缓存**（`find . -name '*.pyc' -delete`）

## 下一步工作方向

1. **已确认 pass**: 3615185 通过安全门 → 运行 3×500 局重复确认
2. **后续**: 更长训练（50k-100k 手）+ 全 13 对手评估 → 冲刺交付标准
3. **中期**: PCGrad 梯度冲突处理、CFR 蒸馏

## 关键文档

- `docs/agent_training_log.md` — 详细实验历史（944行，必读）
- `docs/two_server_seed_rl_plan.md` — 双服务器方案
- `docs/seed_rl_architecture.md` — 代码架构说明
- `docs/research_training_plan.md` — 研究训练计划
