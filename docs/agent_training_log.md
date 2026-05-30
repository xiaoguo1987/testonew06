# Agent Training Log

This file is the running handoff between Codex and DeepSeek for the 6-max
100bb NLHE research model.  Keep user-facing progress here so the project can
be checked without reading terminal scrollback.

## Goal

Deliver a trained 6-max 100bb cash-game research model that can later be used by
the advisor layer for action suggestions and test-environment assistance.

## Current Branch

- `codex/seed-rl-research-20260530`

## Current Definition Of Done

The first deliverable model is considered usable only when all of these are
true:

- A reproducible checkpoint is saved as the selected model artifact.
- Training and evaluation commands are recorded.
- Metrics include speed, PPO update count, KL, action distribution, position
  profit, invalid bet rate, and truncated hand rate.
- Evaluation includes at least random opponents and frozen opponent-model
  opponents.
- Truncated hand rate is below 1% on the main evaluation suite.
- Invalid bet rate is below 1%.
- Action distribution is not dominated by a single shortcut action.  In
  particular, call and all-in rates must be reviewed by street/SPR before the
  checkpoint is accepted.
- The final report explains known limitations before any real-time advisor use.

This is not a claim that the strategy is unexploitable.  Full exploitability or
CFR best-response evaluation is still a future milestone.

## Persistent Server Runbook

Server project directory:

```bash
/root/testo-seed-rl
```

Latest run pointer:

```bash
/root/testo-seed-rl/runs/latest_run_dir
```

Common files inside a run directory:

```bash
train.log
metrics.jsonl
checkpoints/
agent_training_log.md
```

## Experiment History

### 2026-05-30 12-action baseline

- Run: `/root/testo-seed-rl/runs/seed_3m_12act_20260530_1550`
- Hands: 3,000,043
- Speed: about 1,734 hand/s
- Final cumulative action mix: call 56.44%, all-in 14.87%, fold 11.67%.
- Evaluation against opponent model was unstable and often weak after the first
  checkpoint.
- Takeaway: expanded action abstraction worked, but reward shaping and all-in
  incentives still encouraged shortcut strategies.

### 2026-05-30 reward safety smoke

- Run: `/root/testo-seed-rl/runs/seed_reward_safety_smoke_20260530_1625`
- Removed VPIP/courage/raise shaping bonuses.
- Lowered all-in unlucky compensation defaults.
- Added stricter final-buffer handling.
- Takeaway: reward signal became less shortcut-heavy, but all-in still needed a
  strategy gate.

### 2026-05-30 all-in gate validation

- Run: `/root/testo-seed-rl/runs/seed_1_5m_allin_gate_20260530_1645`
- Hands: 1,500,036
- Elapsed: 1,092.2s
- Speed: 1,373.4 hand/s
- PPO updates: 199
- KL: median 0.110, p95 0.706, max 0.856
- Final cumulative action mix:
  - fold 23.20%
  - call 53.61%
  - ordinary raises 13.28%
  - all-in 9.91%
- Final interval action mix:
  - fold 27.99%
  - call 39.27%
  - ordinary raises 20.15%
  - all-in 12.59%
- Evaluation:
  - 500k opponent-model: +100.9 bb/100, truncated 0.1%
  - 1000k opponent-model: -125.3 bb/100, truncated 0.3%
- Takeaway: explicit all-in gate reduced deep-stack shove shortcuts, but
  opponent-model evaluation remains unstable and KL tail risk is too high.

### 2026-05-30 aborted resume stability run

- Run: `/root/testo-seed-rl/runs/seed_stability_200k_from500k_20260530_171036`
- Intended plan: resume from the 500k checkpoint and run 200k stability hands.
- Stopped after the startup diagnostics revealed a resume bookkeeping bug:
  `total_hands` restored to 500,016 while the new run target was 200,000, so
  curriculum progress was computed as 250% and immediately jumped to Phase3.
- Startup evaluation/checkpoint/opponent update also fired immediately because
  the last-eval and last-checkpoint cursors started at 0 instead of the resumed
  hand count.
- Fix: resume runs now treat `num_hands` as additional hands for the current
  stage, initialize interval cursors from the resumed hand count, and calculate
  curriculum progress from this stage's newly generated hands.

## Codex Notes

- Do not optimize toward a fixed all-in frequency.  All-in should be
  conditional on stack depth, SPR, invested chips, fold equity, and hand/board
  equity once those estimates are available.
- Current model quality cannot be trusted from self-play profit alone.  Use
  opponent-model, random/fish, older checkpoints, and eventually
  best-response/CFR-style evaluations.
- Speed should be tracked with both hand/s and inference items/s.  Reducing
  all-in can naturally lower hand/s because hands last longer.

## DeepSeek Notes

### 2026-05-30 next-run review

- Prefer resuming from the 500k checkpoint from the all-in gate validation run,
  because that was the latest checkpoint with positive opponent-model
  evaluation.
- Do not immediately commit to a huge run from a potentially unstable policy.
  Run a 200k stability stage first, then extend if the metrics are clean.
- Use lower learning rate for the stability stage to reduce KL tail risk.
- Save frequently enough to preserve recoverable checkpoints.
- Do not skip the final evaluation when accepting a deliverable model.
- Stop and inspect if opponent-model bb/100 drops below -50, all-in interval
  exceeds 20%, call interval falls below 30%, truncation exceeds 1%, invalid
  bet rate exceeds 1%, or KL tail keeps growing.

## Next Work Queue

- Start a 200k stability run from the 500k checkpoint, then extend only if
  evaluation remains stable.
- Periodically summarize metrics into this file.
- Add richer all-in diagnostics: all-in by street/SPR, all-in called rate, and
  all-in showdown result.
- Add evaluation stability gates before accepting `seed_rl_best`.
- Reduce KL tail risk with PPO clipping/entropy/buffer tuning if instability
  repeats.
