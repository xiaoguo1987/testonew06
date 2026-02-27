# Quickstart

## 1) 安装依赖

```bash
pip install -r requirements.txt
```

## 2) 500 手牌快速验证（PPO）

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

## 3) 评估模型（推荐）

```bash
python train.py --mode eval --model-type ppo \
  --model-path ./checkpoints_run500_fixwinrate/agent_ppo_500 \
  --num-eval-games 500 \
  --eval-output eval_results.json
```

## 4) 从最新检查点继续训练

```bash
python train.py --mode train --model-type ppo --resume \
  --checkpoint-dir ./checkpoints_run500_fixwinrate
```

## 5) 日志与结果位置

- 训练日志：`logs/`
- 检查点：`checkpoints*/`
- 训练总结：`checkpoints*/training_summary_*.json`
