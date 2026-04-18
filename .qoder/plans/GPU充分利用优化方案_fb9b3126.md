# RTX 4090D GPU 充分利用优化方案

## 问题诊断

当前训练流程的时间分配：
- **CPU自我对弈采样**: ~99%（16个线程模拟牌局、Python逻辑、NumPy状态编码）
- **GPU PPO参数更新**: ~1%（256 batch, 4 epochs, 几十毫秒完成）

**核心矛盾**：4090D 有 24GB 显存、82 TFLOPS 算力，但模型只有 ~200K 参数、batch=256，GPU 几毫秒就做完更新然后长时间空闲等数据。

## 优化方案（3个维度，按收益排序）

### 维度1: 增大GPU计算量（收益最大，改动最小）

**原理**：让GPU每次更新有更多工作要做，减少空闲占比。

**1.1 升级网络架构** — `ppo_model.py`
- 当前: 256-256-128-64（约200K参数）
- 优化: **512-512-256-128**（约800K参数），4090D轻松承载
- 同时增大残差连接维度匹配
- 估算：训练耗时从~50ms增加到~200ms，GPU利用率提升4x

**1.2 增大训练batch** — `ppo_config.py` + `train.py`
- 当前: batch_size=256, ppo_update_min_steps=自动(~2048)
- 优化: **batch_size=1024**, ppo_update_min_steps=**8192**
- 更大批次让GPU有更多矩阵运算，且PPO训练更稳定

**1.3 增加update_epochs** — `ppo_config.py`
- 当前: update_epochs=4
- 优化: **update_epochs=8**（配合KL早停，不会导致策略退化）
- 同一批数据多训练几轮，GPU多做几次前向/反向传播

### 维度2: GPU参与采样阶段（收益中等，改动适中）

**原理**：当前16个worker各自持有一个CPU上的模型副本做推理（`clone_for_inference`），GPU完全空闲。改为在主线程用GPU做批量推理。

**2.1 主线程GPU批量推理** — `self_play_trainer.py`
- 当前流程: worker线程 → CPU副本推理 → 收集轨迹
- 优化流程: worker线程 → 只做环境模拟（不推理）→ 收集所有待决策状态 → **主线程GPU批量推理** → 返回动作
- 具体改动：
  - `_play_hands_collect_batched`中，不再调用`agent_copy.select_actions_batch_with_info`
  - 改为收集`main_states`列表，通过共享队列传给主线程
  - 主线程用`self.main_agent.agent.predict_batch(all_states)`做GPU批量推理
  - 推理结果通过队列返回给worker

**2.2 移除clone_for_inference** — `ai_agent.py` + `self_play_trainer.py`
- 当前每个worker克隆一个完整PPO模型（含Actor+Critic），占大量CPU内存
- GPU批量推理后，worker不再需要模型副本，只做纯环境模拟
- 节省CPU内存，同时释放CPU算力给环境模拟

### 维度3: 异步流水线训练（收益显著，改动较大）

**原理**：当前采样和训练是串行的（采样完→训练→采样完→训练...），GPU训练时CPU空闲，CPU采样时GPU空闲。改为异步流水线，让采样和训练重叠执行。

**3.1 训练线程** — `self_play_trainer.py`
- 启动一个后台训练线程，从共享buffer中取数据做PPO更新
- 主线程持续采样，不阻塞等待训练完成
- 使用`threading.Thread(daemon=True)` + `queue.Queue`
- 关键：权重同步需要在训练完成后加锁更新到推理副本

**3.2 双缓冲机制**
- Buffer A：当前正在被训练线程使用的轨迹数据
- Buffer B：主线程正在写入的新轨迹数据
- 训练完成后交换A/B，避免数据竞争

## 文件改动清单

| 文件 | 改动内容 | 维度 |
|------|---------|------|
| `poker_ai/model/ppo_model.py` | 升级网络架构到512-512-256-128 | 1 |
| `poker_ai/config/ppo_config.py` | batch_size=1024, update_epochs=8, ppo_update_min_steps=8192 | 1 |
| `train.py` | rtx4090d profile更新batch_size/ppo_update_min_steps | 1 |
| `poker_ai/reward/self_play_trainer.py` | GPU批量推理+异步训练线程+双缓冲 | 2,3 |
| `poker_ai/model/ai_agent.py` | 移除clone_for_inference中的PPO模型复制 | 2 |

## 预期效果

| 指标 | 当前 | 优化后 |
|------|------|--------|
| GPU利用率 | 0-1% | 30-60% |
| GPU显存占用 | ~200MB | ~2-4GB |
| PPO训练耗时/次 | ~50ms | ~500ms-1s |
| 训练总速度 | ~X hands/s | ~1.5-2X hands/s |
| 模型参数量 | ~200K | ~800K |

## 实施优先级

1. **维度1**（改动小，立即可做）：升级模型+增大batch → GPU利用率提升约4-8x
2. **维度2**（改动中等）：GPU批量推理 → GPU利用率再提升2-3x
3. **维度3**（改动大，可选）：异步流水线 → 整体训练速度提升1.5-2x