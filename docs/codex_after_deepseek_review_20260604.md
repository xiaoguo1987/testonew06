# Codex Work Log After DeepSeek Review

记录范围：从用户问“DeepSeek 看了代码没问题，这个还需要跑 3M 手看效果吗”之后开始，到本次提交前为止。

## 结论

不建议直接跑 3M 手。原因是当前 164 维模型虽然均值指标有希望，但历史结果已经显示 robustness/stability 是主要短板；如果直接把不稳定 checkpoint 放大到 3M，容易得到均值好看但最差对手崩盘的模型。

本轮采用分阶段验证：

1. 先从 164 维 best mean checkpoint `seed_rl_hands_1000194` 跑 50k smoke。
2. 训练结束后先做 hard-subset repeat validation。
3. 如果 hard subset 稳定，再考虑 300k/500k。
4. 只有 1M 级别通过 repeated gate，才值得跑 3M。

本轮结果：50k smoke 机械上成功，但策略漂移和行级波动都太大，不能晋级到 300k，更不能直接跑 3M。

## 训练记录

运行目录：

`/root/testo-seed-rl/runs/seed_164_profile_smoke_50k_from1000k_20260604_215444`

基础模型：

`/root/testo-seed-rl/runs/seed_164dim_3M_20260604_170910/checkpoints/seed_rl_hands_1000194`

主要参数：

- hands: `50,000`
- learning rate: `3e-5`
- checkpoint interval: `25,000`
- eval interval: `50,000`
- eval games: `100`
- curriculum: Phase 3 only
- opponent roster: `configs/seed_roster_3600064_weighted_safety.json`
- opponent ensemble size: `3`
- skip final buffer train: enabled

训练完成：

- final checkpoint: `seed_rl_hands_1050210`
- total hands: `1,050,210`
- PPO updates: `14`
- elapsed: `155.5s`
- speed: `321.6 hands/s`
- C++ engine: active, `164-dim`
- invalid/truncated in quick eval: `0`

## 训练问题

`run_report.md` 显示 PPO policy drift 过大：

- KL median: `0.3846`
- KL p95: `0.4536`
- KL max: `0.4666`
- cumulative action distribution: `call=74.67%`
- last interval action distribution: `call=80.16%`

这个动作分布明显偏向 call-heavy，说明 `3e-5` 对这个 continuation point 偏激进。该 checkpoint 不适合继续放大训练。

训练自带 100 局 quick eval 波动也很大：

- mid eval: `+596.2 bb/100`
- final eval: `-63.0 bb/100`

100 局样本不能作为晋级依据，因此又跑了 hard-subset repeat validation。

## Hard-Subset 验证

验证对象：

`seed_rl_hands_1050210`

验证目录：

`/root/testo-seed-rl/runs/seed_164_profile_smoke_50k_from1000k_20260604_215444/safety_hard6_repeats2_300`

验证设置：

- hard opponents: `1000262`, `1500036`, `600040`, `1000054`, `600047`, `700169`
- games: `300`
- repeats: `2`
- invalid gate: `0.001`
- truncated gate: `0.005`

原脚本聚合结果给了 `PASS`，但实际 row-level 稳定性不合格：

| opponent | mean bb/100 | min repeat | max repeat | std |
|---|---:|---:|---:|---:|
| `seed_rl_hands_600047` | `-295.8` | `-585.4` | `-6.3` | `409.5` |
| `seed_rl_hands_600040` | `-184.6` | `-588.0` | `218.8` | `570.5` |
| `seed_rl_hands_1000262` | `145.8` | `-317.4` | `609.1` | `655.1` |
| `seed_rl_hands_1000054` | `317.8` | `194.8` | `440.8` | `174.0` |
| `seed_rl_hands_1500036` | `440.8` | `-22.6` | `904.3` | `655.5` |
| `seed_rl_hands_700169` | `1039.4` | `746.9` | `1332.0` | `413.7` |

结论：该 checkpoint 不能晋级。均值并不差，但最差 repeat 和 row std 说明策略不稳定。

## 本次代码改动

### `scripts/seed_safety_validate.py`

新增两个门禁参数：

- `--min-row-min`
  - repeat validation 时，如果某个 opponent 的最低 repeat 分数低于该阈值，判失败。
- `--max-row-std`
  - repeat validation 时，如果某个 opponent 的 row std 超过该阈值，判失败。

目的：防止类似 `seed_rl_hands_1050210` 这种“聚合均值过关，但单次 repeat 大崩盘”的 checkpoint 被误判为 PASS。

### `scripts/run_seed_reference_smoke.sh`

新增环境变量并传给 `seed_safety_validate.py`：

- `VALIDATION_MIN_ROW_MIN`
  - default: `-500`
- `VALIDATION_MAX_ROW_STD`
  - default: `300`

目的：训练阶段 smoke gate 直接使用行级稳定性门槛，避免继续放大高波动候选。

### `docs/agent_training_log.md`

追加本次 50k smoke、hard-subset repeat validation 和 gate 修复结论。

## 硬编码问题

用户指出“很多代码硬编码不太好”，这个判断是对的。本轮没有全面重构硬编码，只修补了最影响验收正确性的 gate。后续建议 DeepSeek 或 Codex 优先整理这些点：

- `scripts/run_seed_reference_smoke.sh` 默认写死了：
  - `ROOT_DIR=/root/testo-seed-rl`
  - `PYTHON=/root/miniconda3/bin/python`
  - `OPPONENT_NAMES=...`
  - `OPPONENT_ROSTER_FILE=...`
- hard-subset 对手列表目前偏实验脚本配置，不是统一 config。
- gate threshold 分散在 bash defaults 和 Python parser defaults 中，建议统一放到 JSON/YAML config。
- 多个脚本仍通过 shell 拼接命令，长期看应改成 Python orchestration 或明确的 config-driven launcher。
- 服务器路径、run naming、checkpoint resolver 都应配置化，避免换服务器时重复改脚本。

## 建议下一步

不要从 `seed_rl_hands_1050210` 继续训练。

建议从 `seed_rl_hands_1000194` 重新做更保守的 smoke：

- LR 降到 `1e-5` 或 `5e-6`
- 显式使用更严格的 row-level gate
- 加强 KL/anchor/reference 约束
- 先跑 `50k`
- 通过 hard-subset `2x300` 或 `3x300` 后再放大到 `300k`

如果交给 DeepSeek 继续，建议它先处理 config 化和 gate 统一，再跑新训练。
