# 164 维状态升级总结

## 日期：2026-06-04

## 改动概览

### 状态维度扩展：134 → 140 → 164

| 阶段 | 维度 | 新增内容 |
|------|------|---------|
| 原始 | 134 | 游戏状态（手牌/公共牌/玩家信息/位置/活跃人数） |
| Day2 | 140 | +6 对手类型标识（每座位：0=鱼，1=PPO） |
| Day2 | 164 | +24 对手行为 EMA（每座位 4 统计：VPIP/PFR/AF/fold_to_3bet） |

### 修改的文件

| 文件 | 改动 |
|------|------|
| `poker_ai/model/state_representation.py` | 新增 `opponent_type_dim=6`、`opponent_behavior_dim=24`、`get_opponent_type_vector()`、`get_opponent_behavior_vector()` |
| `poker_ai/config/ppo_config.py` | `input_dim`: 134→140→164 |
| `poker_ai/training/env_worker.py` | 牌桌级对手统计跟踪（EMA decay=0.99）、`_update_opp_stats()`、`_update_opp_stats_cc()`、`_set_opp_features_cc()` |
| `poker_ai/training/gpu_trainer.py` | 评估时传递对手类型 |
| `poker_ai/model/large_ppo_model.py` | 自动权重扩展（134→140、140→164）、PCGrad-lite 时间分片聚类 |
| `poker_ai/engine/poker_engine.cpp` | C++ 状态向量 134→140→164 |

## 6 对手硬子集最佳结果

| 检查点 | 维度 | 手数 | avg | worst | 对手 |
|--------|------|------|-----|-------|------|
| 3615133 | 134 | 15k | -71.0 | -375.5 | 1500036 |
| 3680243 | 140 | 80k | +126.8 | -99.2 | 1000054 |
| 3650085 | 140 | 50k | +78.1 | -405.7 | 700169 |
| 3000047 | 140 | 3M(fresh) | +140.6 | -123.9 | 1000054 |
| 1000194 | 164 | 1M(fresh) | +105.3 | -245.3 | 1500036 |
| 3000026 | 164 | 3M(fresh) | +132.2 | -192.0 | 1000054 |

## 13 对手全量验收最佳结果

| 检查点 | 维度 | avg | worst | max_std | 状态 |
|--------|------|-----|-------|---------|------|
| 3680243 | 140 | +29.1 | -443.0 | 779.8 | ❌ |
| 3000047 | 140 | +35.4 | -220.7 | 354.9 | ❌ 差 21 |
| 1000194 | 164 | +100.6 | -251.1 | 529.5 | ❌ 差 51 |
| 3000026 | 164 | +0.8 | -235.0 | 340.8 | ❌ 差 35 |

## 核心发现

1. **140 维对手身份特征**将 avg 从 -71 提升到 +35~140（6对手），显著改善
2. **164 维行为统计**将 avg 进一步提升到 +100.6（13对手），但 worst-case 未改善
3. **多目标梯度冲突是根本瓶颈**：每次 PPO 更新改善某个对手时弱化另一个
4. **更长训练不收敛**：3M vs 1M，avg 反而从 +100 降到 +0.8
5. **PCGrad 时间分片**因单缓冲对手多样性不足而未激活冲突检测
6. **针对性修复**反复造成灾难性遗忘

## 建议后续方向

1. **轨迹级对手标签 + 真正 PCGrad**：修改数据管线，在梯度层面解决冲突
2. **多模型集成**：不同检查点对不同对手表现好，取最优组合
3. **CFR 蒸馏**：用博弈论方法替代 PPO 自对弈

## Codex 复查修复（2026-06-04）

- 评估路径现在维护并传入与训练一致的 24 维对手行为 EMA，避免 164 维模型评估时后 24 维全为 0。
- 对手行为统计改为每手 flags：VPIP/PFR 只看翻牌前自愿入池/加注，AF 使用 aggressive/(aggressive+passive)，fold-to-3bet 只在实际面对 3bet 时更新。
- PPO 轨迹增加 opponent group 标签（对手类型 + ensemble checkpoint basename），训练日志会输出本次 buffer 的样本组分布。
- `LargePPOModel.train()` 支持按显式 opponent group 做 minibatch 聚类；这只是后续真正 per-opponent PCGrad/梯度投影的前置能力，不应当视作完整 PCGrad 已完成。
- VPIP/PFR/AF/fold-to-3bet 的统计规则已抽到 `poker_ai/training/opponent_profile.py`，训练 worker（Python/C++）和评估路径共用同一套 helper，避免训练/评估画像语义漂移。
