# 6-Max Cash Research Training Plan

## Goal

Train a research-focused 6-player no-limit hold'em cash-game advisor model for 100BB stacks. The first milestone is a fast, stable self-play model that can produce action probabilities, a recommended action, value estimates, and diagnostics. Opponent-specific adaptation comes later through collected player features.

## Current Baseline

This branch uses the high-throughput SEED RL path:

- `train.py --arch seed-rl`
- `LargePPOModel`
- C++ environment engine when available
- centralized GPU batch inference
- vectorized CPU workers with many tables per worker
- curriculum opponents and a historical opponent pool

The legacy Python `SelfPlayTrainer` remains available through `--arch legacy`, but it is not the target path for long runs.

## Initial Game Assumptions

- Game type: 6-max cash
- Stack: 100BB
- Default blinds: SB 5, BB 10, stack 1000
- Initial abstract actions: fold, call, small raise, mid raise, large raise, overbet, all-in
- Primary metric: bb/100 with confidence from repeated evaluations
- Safety metric: invalid bet rate must remain near zero

## Theory References

- Pluribus: multi-player no-limit hold'em self-play and search ideas.
  https://www.science.org/doi/10.1126/science.aay2400
- DeepStack: continual re-solving for imperfect-information poker.
  https://arxiv.org/abs/1701.01724
- Libratus: nested subgame solving and self-improvement.
  https://www.ijcai.org/proceedings/2017/0772.pdf
- ReBeL: self-play plus search in imperfect-information games.
  https://arxiv.org/abs/2007.13544
- NFSP: average strategy and best-response style baseline.
  https://arxiv.org/abs/1603.01121

## Near-Term Training Protocol

1. Smoke test: 1k to 10k hands, `--skip-final-eval`, verify C++ engine and metrics output.
2. Throughput baseline: 100k hands, no frequent eval, target high hands/s.
3. Stability run: 1M hands, checkpoint every 50k, evaluation every 50k.
4. Quality run: repeated evaluation against random, fish, older checkpoints, and frozen best models.

## Required Logs

Every training run writes normal logs plus `metrics.jsonl` in the checkpoint directory. Important events:

- `train_start`: hardware/concurrency/game configuration
- `ppo_update`: hands/s, interval hands/s, PPO time, KL, loss, inference batch statistics
- `evaluation`: win rate, avg profit, bb/100, invalid bet rate, action distribution
- `train_complete`: total speed, updates, best score
- `final_evaluation`: final quality snapshot

These records are intended for Codex, Doubao, and DeepSeek review.

## Example Fast Baseline

```bash
python train.py \
  --mode train \
  --hardware-profile rtx4090d \
  --arch seed-rl \
  --num-hands 100000 \
  --checkpoint-dir runs/seed_rl_100k/checkpoints \
  --metrics-output runs/seed_rl_100k/metrics.jsonl \
  --eval-output runs/seed_rl_100k/eval_results.json \
  --checkpoint-interval 50000 \
  --eval-interval 50000 \
  --num-eval-games 100 \
  --skip-final-eval
```

## Agent Review

Set environment variables before review:

- `ARK_API_KEY`
- `DEEPSEEK_API_KEY`
- optional `DOUBAO_MODEL`
- optional `DEEPSEEK_MODEL`

Then run:

```bash
python tools/agent_review.py
```
