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

### 2026-05-30 stopped fixed resume stability run

- Run: `/root/testo-seed-rl/runs/seed_stability_200k_from500k_fixed_20260530_171425`
- Stopped at the first evaluation:
  - hands 600,291
  - opponent-model bb/100 -47.3
  - invalid bet rate 0.58%
  - truncated hand rate 1.10%
  - KL median 0.0945, p95 0.1816, max 0.2235
  - cumulative action mix: call 51.86%, all-in 10.87%
- Takeaway: resume accounting is fixed and PPO stability is acceptable, but
  opponent-model evaluation is still too brittle.  The likely cause is
  deterministic argmax evaluation against a mirrored/stale model, which can
  create repeated low-pressure check/call states.
- Fix in progress: load historical checkpoints from the external source run into
  the opponent pool, avoid phase-switch mirror sync when history exists, and use
  low-temperature model-action sampling for opponent-model evaluation.

### 2026-05-30 historical-pool evaluation smoke

- Run: `/root/testo-seed-rl/runs/seed_eval_pool_smoke_20260530_172039`
- Purpose: validate historical opponent pool loading and low-temperature
  opponent-model evaluation.
- Result:
  - external checkpoint directory contributed 3 historical opponents
  - truncation dropped to 0.00% in all evaluation events
  - 510k eval vs selected historical opponent: +99.4 bb/100
  - 520k eval after Phase3 loaded `seed_rl_hands_1500036`: -351.4 bb/100
  - final eval: -349.5 bb/100
- Takeaway: evaluation-loop truncation is fixed, but the 500k policy is weak
  against later historical opponents.  Continue with longer historical-pool
  training and select the best checkpoint by multi-opponent evaluation, not by
  fish/random profit.

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

### 2026-05-30 historical-pool long-run review

- After the historical-pool smoke test, Codex and DeepSeek agree the immediate
  blocker is not raw speed or all-in gating; it is robustness against stronger
  historical checkpoints.
- Resume from the 500k checkpoint rather than the 1.5M terminal checkpoint,
  because 500k was the most stable known policy and the 1.5M policy is already
  a hard opponent in the pool.
- Run historical-pool training in stages.  First extend by roughly 1M hands with
  low learning rate and 100k checkpoints, then expand toward 10M+ only if
  multi-opponent evaluation improves.
- Continue if truncation stays below 1%, invalid bet rate stays below 1%, no
  NaN/infinite losses appear, all-in stays roughly below 18-20%, and historical
  opponent bb/100 improves or does not degrade across two evaluation windows.
- Roll back to the prior good checkpoint and reduce learning rate further if
  strong-history bb/100 remains worse than about -100 after another 400k hands,
  or if all-in spikes above 20%.
- Next code gap: add independent per-historical-opponent evaluation instead of
  relying on one selected opponent-model evaluation.

### 2026-05-30 stopped Phase1-biased historical-pool run

- Run: `/root/testo-seed-rl/runs/seed_histpool_1m_from500k_20260530_230835`
- Stopped after the first evaluation because the policy was moving in the wrong
  direction:
  - hands 700,351
  - opponent-model bb/100 -495.5
  - win rate 17.9%
  - invalid bet rate 0.00%
  - truncated hand rate 0.00%
  - KL median 0.0289, p95 0.1830, max 0.2169
  - cumulative action mix: call 50.98%, all-in 12.67%
  - median interval speed 1,823 hand/s
- Diagnosis: training mechanics were stable, but the resumed 500k checkpoint
  re-entered Phase1 because curriculum progress was measured from the new
  stage start.  That means the first 200k hands were mostly against fish, which
  is the wrong distribution for historical-pool robustness.
- Fix: add a minimum curriculum start phase.  Historical-pool continuation can
  now start at Phase2 and immediately load an opponent from the historical pool.

### 2026-05-30 Phase2 historical-pool restart checkpoint

- Run: `/root/testo-seed-rl/runs/seed_histpool_phase2_1m_from500k_20260530_231522`
- Startup confirmed:
  - `curriculum_start_phase=2`
  - `min_phase=Phase2`
  - historical opponent loaded at startup: `seed_rl_hands_1000054`
- First evaluation improved materially over the Phase1-biased run:
  - 700,016 hands: win rate 19.2%, bb/100 -101.6, invalid 0.05%,
    truncated 0.10%
- Second evaluation was profitable but had low hand win rate:
  - 900,022 hands: win rate 18.7%, bb/100 +460.1, invalid 0.00%,
    truncated 0.00%
- Diagnosis: the old safety valve only tracked win rate, so it froze opponent
  updates even when bb/100 was positive.  For poker training this is wrong
  because a strategy can lose more hands than it wins while still winning chips.
- Fix: safety-valve low-win streak now requires both low win rate and negative
  bb/100; positive bb/100 clears the streak.
- Follow-up fix: opponent-pool sampling now avoids selecting the exact base
  checkpoint being resumed from when other historical checkpoints are available.

### 2026-05-30 non-self continuation stopped

- Run: `/root/testo-seed-rl/runs/seed_histpool_nonself_600k_from900k_20260530_232606`
- Startup confirmed the non-self opponent-pool fix:
  - base checkpoint: `seed_rl_hands_900022`
  - startup opponent: `seed_rl_hands_700169`
- Stopped after the first evaluation:
  - 1,100,057 hands: win rate 16.7%, bb/100 -309.0, invalid 0.00%,
    truncated 0.00%
  - KL median 0.1246, p95 0.2057, max 0.2702
- Diagnosis: the 900k checkpoint is not uniformly robust across historical
  opponents.  Continue only after running a multi-opponent evaluation matrix to
  identify which checkpoints are genuine improvements and which opponents are
  causing collapse.
- Added `scripts/evaluate_seed_pool.py` for explicit checkpoint-vs-checkpoint
  evaluation.

### 2026-05-30 checkpoint matrix comparison

- 900k matrix, 500 games/opponent:
  - average approximately -150 bb/100
  - large losses vs `seed_rl_hands_500016`, `seed_rl_hands_600040`,
    `seed_rl_hands_700169`, and `seed_rl_hands_900236`
  - conclusion: discard 900k as a base checkpoint for now
- 500k matrix, 500 games/opponent:
  - average approximately +15 bb/100
  - large losses vs `seed_rl_hands_1000054` and `seed_rl_hands_700016`
  - conclusion: 500k is the better restart base, but it needs targeted hard
    opponent training
- DeepSeek agrees: continue from 500k, reduce LR to about `2e-5`, evaluate
  frequently, and accept a new base only if multi-opponent average bb/100 is
  above +10 and worst opponent is no worse than roughly -200 bb/100.
- Fix: hard-opponent weighted sampling added.  Training can now pass a matrix
  JSONL file and sample opponents by softmax of recent negative bb/100.

### 2026-05-30 hard-opponent 100k stage result

- Run: `/root/testo-seed-rl/runs/seed_hardopp_100k_from500k_20260530_234418`
- Setup: base 500k, LR `2e-5`, Phase2 start, hard-opponent score file from
  the 500k matrix.
- Startup sampled the hardest scored opponent:
  `seed_rl_hands_1000054` with score -673.8 bb/100.
- Training result:
  - 550,077 eval vs current opponent: -365.3 bb/100
  - 600,042 final eval vs current opponent: +42.3 bb/100
  - KL median 0.1002, p95 0.1765, max 0.1833
- Matrix result did not pass acceptance:
  - `1000054` improved from about -674 to -258 bb/100
  - but `700016`, `800200`, and self-adjacent checkpoints regressed heavily
  - average moved negative, roughly -95 bb/100
- Diagnosis: the hard-opponent focus helped one weak matchup but the short run
  switched into Phase3 at 50k and caused broader drift.
- Fix: add `curriculum_max_phase` so short repair stages can stay locked in
  Phase2.

### 2026-05-31 Phase2-locked repair comparison

- Run: `/root/testo-seed-rl/runs/seed_hardopp_phase2_100k_from500k_20260530_235204`
- Setup: base 500k, LR `2e-5`, `CURRICULUM_START_PHASE=2`,
  `CURRICULUM_MAX_PHASE=2`, hard-opponent temperature 500.
- Result:
  - 550,046 eval: +201.7 bb/100
  - 600,047 eval: +185.4 bb/100
  - matrix average: about +71 bb/100
  - worst opponent: `seed_rl_hands_1000054`, about -635 bb/100
- Takeaway: this is the best candidate so far by matrix average, but it still
  fails the acceptance rule because the worst matchup remains below -200.

- Follow-up run: `/root/testo-seed-rl/runs/seed_hardopp_phase2_100k_from600k_20260530_235757`
- Setup: base 600,047, LR `1e-5`, Phase2 locked, score file refreshed from
  the 600k matrix.
- Result:
  - 650,056 eval: -45.7 bb/100
  - 700,057 eval: -27.1 bb/100
  - matrix average: about -46 bb/100
- Takeaway: do not use 700,057 as a base.  Keep 600,047 as the current best
  candidate while continuing targeted fixes for `seed_rl_hands_1000054`.

### 2026-05-31 hard-opponent pool loading fix

- Heartbeat check found that hard-opponent weighted sampling loaded scores but
  did not automatically add the scored checkpoint paths to the training pool.
- Impact: several matrix opponents could be weighted in theory but never sampled
  in practice unless they happened to live in the current or base checkpoint
  directory.
- Fix: score-file opponent paths are now inserted into `_opponent_pool` during
  training startup, and startup logs report `score_file+N`.
- Validation run:
  `/root/testo-seed-rl/runs/seed_hardopp_poolfix_50k_from600k_20260531_001222`
  loaded `score_file+10` and sampled matrix opponent `seed_rl_hands_700016`,
  confirming the pool loading fix works.
- Validation matrix:
  `/root/testo-seed-rl/runs/seed_pool_eval_650k_poolfix_20260531_001445`
  did not pass acceptance.  It improved `seed_rl_hands_1000054` to positive,
  but regressed `seed_rl_hands_1500036` and `seed_rl_hands_1000262`; average was
  roughly -11 bb/100.
- Follow-up fix: when `OPPONENT_SCORE_FILE` is active, the dynamic opponent
  interval is now capped by `OPPONENT_UPDATE_INTERVAL`.  Short repair stages can
  rotate through multiple scored hard opponents instead of staying on one
  sampled opponent for the whole Phase2 stage.
- Current best candidate remains `seed_rl_hands_600047` from
  `/root/testo-seed-rl/runs/seed_hardopp_phase2_100k_from500k_20260530_235204`.
  Do not continue from 650k without a better multi-target weighting scheme.

### 2026-05-31 3M self-play sweep from 600k

- Run: `/root/testo-seed-rl/runs/seed_selfplay_3m_from600k_20260531_093252`
- Setup: base `seed_rl_hands_600047`, 3M additional hands, LR `2e-6`,
  checkpoint/eval every 100k, Phase2 start and Phase3 allowed.
- Training completed at `seed_rl_hands_3600064`.
  - Overall speed: 973.4 hands/s including startup, checkpoints, and eval.
  - Median interval speed: 1424.2 hands/s.
  - KL median/p95/max: 0.1290 / 0.2417 / 0.3464.
  - Final cumulative actions: fold 21.9%, call 57.8%, all-in 8.0%.
  - Phase3 online evals were noisy: `2702798` +420.5 bb/100,
    `2902893` +360.2 bb/100, `3202958` -521.4 bb/100, final `3600064`
    +152.8 bb/100.
- Candidate sweep:
  `/root/testo-seed-rl/runs/seed_pool_sweep_3m_candidates_20260531_102722`
  against the 13-opponent historical pool, 500 games/opponent.
  - `600047`: avg +24.8, worst `600049` -1080.1.
  - `800093`: avg -88.0, worst `1000262` -650.8.
  - `1400635`: avg +22.7, worst `1000054` -919.6.
  - `1902135`: avg -172.7, worst `600049` -603.1.
  - `2702798`: avg -146.0, worst `1500036` -1035.6.
  - `2902893`: avg -215.7, worst `700169` -867.4.
  - `3503107`: avg +49.8, worst `600049` -559.2, then
    `600047` -524.0 and `1000054` -369.8.
  - `3600064`: avg -143.6, worst `600040` -567.0.
- Takeaway: use `3503107` as the next research base because it has the best
  matrix average and improves the worst-case relative to `600047`, but it is
  not acceptable for delivery.  The next stage should target worst-matchup
  robustness, not raw online eval.
- Started follow-up run:
  `/root/testo-seed-rl/runs/seed_robust_repair_100k_from3503k_20260531_105520`
  from `3503107`, LR `1e-6`, Phase2 locked, score-file from the `3503107`
  matrix, hard-opponent temperature 1000, opponent rotation every 10k.

### 2026-05-31 hard-negative repair sampling

- The first 100k robust repair from `3503107` did not pass matrix validation:
  - Run:
    `/root/testo-seed-rl/runs/seed_robust_repair_100k_from3503k_20260531_105520`
  - Matrix:
    `/root/testo-seed-rl/runs/seed_pool_eval_robust_repair_20260531_110128`
  - `3555609`: avg -146.2, worst `1000054` -639.4.
  - `3603149`: avg -154.0, worst `800200` -566.2.
- Diagnosis: `hard_opponent_temperature=1000` made score-file softmax too flat,
  so repair sampling was not reliably focused on opponents that beat the current
  model.  DeepSeek review agreed that the next test should explicitly bias
  sampling toward negative bb/100 opponents and use a lower temperature.
- Code change: added `--hard-opponent-negative-ratio`.  When active, each
  hard-opponent refresh samples from score-file opponents with negative bb/100
  with the configured probability, then falls back to the full scored pool for
  the remaining probability.  Default is 0, so previous behavior is unchanged.
- Focused repair validation:
  `/root/testo-seed-rl/runs/seed_focused_repair_100k_from3503k_20260531_111209`
  from `3503107`, LR `1e-6`, Phase2 locked, temperature 200,
  negative-ratio 0.85.
  - Training speed: overall 952.9 hands/s, median interval 1522.5 hands/s.
  - KL median/p95/max: 0.0684 / 0.0878 / 0.0905.
  - Sampling logs confirmed focus on `600049`, `600047`, `1000054`, and
    `900022`.
  - Matrix:
    `/root/testo-seed-rl/runs/seed_pool_eval_focused_repair_20260531_111530`
  - `3553125`: avg -78.0, worst `500016` -708.4.
  - `3603118`: avg +56.3, worst `600047` -521.4.
- Takeaway: `3603118` is a small improvement over `3503107` by matrix average
  and worst-case, but not enough for delivery.  The 50k checkpoint shows that
  over-focusing hard negatives can damage retention versus other historical
  opponents, so the next run should use a lower negative ratio and a slightly
  softer temperature.
- Started follow-up balanced repair:
  `/root/testo-seed-rl/runs/seed_balanced_repair_200k_from3603k_20260531_112308`
  from `3603118`, LR `8e-7`, Phase2 locked, score-file from the `3603118`
  matrix, temperature 250, negative-ratio 0.65.
- Balanced repair completed:
  - Overall speed: 860.6 hands/s, median interval 1510.7 hands/s.
  - KL median/p95/max: 0.0772 / 0.1208 / 0.1312.
  - Online eval was noisy and not predictive: `3653159` -300.3 bb/100,
    `3703192` -181.1, `3753230` +261.6, final `3803147` -21.6.
  - Matrix:
    `/root/testo-seed-rl/runs/seed_pool_eval_balanced_repair_20260531_112838`
  - `3653159`: avg -160.9, worst `900236` -672.6.
  - `3703192`: avg +120.6, worst `1500036` -671.1; high average but too
    brittle.
  - `3753230`: avg -149.5, worst `800200` -525.3; online-best was a false
    positive.
  - `3803147`: avg +103.9, worst `1000262` -360.1; invalid 0.01%, trunc
    0.03%.
- Current best candidate is now `seed_rl_hands_3803147`.  It improves both
  average and worst-case versus `3603118` and `3503107`, but still misses the
  delivery target because worst matchup is below -200 bb/100.
- Started follow-up worst-case repair:
  `/root/testo-seed-rl/runs/seed_worstcase_repair_150k_from3803k_20260531_114355`
  from `3803147`, LR `5e-7`, Phase2 locked, score-file from the `3803147`
  matrix, temperature 180, negative-ratio 0.75.
- Worst-case repair did not pass:
  - Matrix:
    `/root/testo-seed-rl/runs/seed_pool_eval_worstcase_repair_20260531_114747`
  - `3853196`: avg -105.2, worst `700016` -554.2.
  - `3903225`: avg -84.8, worst `550046` -911.8.
  - `3953186`: avg +94.2, worst `800200` -591.8.
  - Takeaway: negative-ratio 0.75 is too aggressive from `3803147`; it repairs
    some sampled weaknesses but creates worse retention failures.  Keep
    `3803147` as current best.
- Started milder repair:
  `/root/testo-seed-rl/runs/seed_mild_repair_100k_from3803k_20260531_115843`
  from `3803147`, LR `3e-7`, Phase2 locked, score-file from the `3803147`
  matrix, temperature 300, negative-ratio 0.50.
- Mild repair did not pass:
  - Matrix:
    `/root/testo-seed-rl/runs/seed_pool_eval_mild_repair_20260531_120313`
  - `3853167`: avg -165.7, worst `900236` -728.3.
  - `3903167`: avg +84.5, worst `1500036` -384.4.
  - `3903174`: avg -7.8, worst `1500036` -527.5.
  - Takeaway: even lower negative-ratio can regress when fallback samples
    unscored intermediate checkpoints from the base run directory.
- Code change: added `--hard-opponent-scored-only-ratio`.  It defaults to 0
  for backwards compatibility.  When enabled, fallback sampling is constrained
  to checkpoint names present in the matrix score file, keeping repair stages
  inside the evaluated 13-opponent population.
- Started scored-only repair:
  `/root/testo-seed-rl/runs/seed_scoredonly_repair_100k_from3803k_20260531_121603`
  from `3803147`, LR `3e-7`, Phase2 locked, score-file from the `3803147`
  matrix, temperature 280, negative-ratio 0.55, scored-only-ratio 1.0.
- Scored-only repair did not pass:
  - Matrix:
    `/root/testo-seed-rl/runs/seed_pool_eval_scoredonly_repair_20260531_122040`
  - `3853168`: avg +77.4, worst `600049` -472.8.
  - `3903168`: avg +103.2, worst `600040` -1086.7.
  - `3903176`: avg +19.0, worst `600047` -612.9.
  - Takeaway: constraining fallback to scored opponents fixes pool cleanliness
    but is not enough by itself.  New repair stages can still create a large
    single-opponent regression, so acceptance needs an explicit retention gate
    before more training is launched.
- Added `scripts/seed_matrix_gate.py` to turn matrix JSONL files into explicit
  PASS/FAIL reports.  Default gates are tuned to the current best model:
  avg >= +103.9 bb/100, worst >= -360.1 bb/100, no fail-fast opponent below
  -500 bb/100, invalid <= 0.1%, trunc <= 0.5%, and 13 evaluated opponents.
- Validation gate report:
  `/root/testo-seed-rl/runs/seed_retention_gate_20260531_143519/gate_report.md`
  compared all worstcase, mild, and scored-only repair checkpoints against
  `3803147`; no candidate passed.  This confirms `3803147` should remain the
  current best until a different training method clears the gate.
- Added fail-fast row gates to `scripts/evaluate_seed_pool.py`.  Matrix sweeps
  can now stop early on a severe per-opponent loss or invalid/truncation spike,
  while still writing the failing row into the JSONL for later inspection.
  Suggested current command flags:
  `--fail-fast-worst-bb100 -500 --max-invalid-rate 0.001 --max-truncated-rate 0.005`.
- DeepSeek reviewed the gate direction and agreed it is the right guardrail
  before more training, but recommended also checking per-opponent retention
  versus the baseline.  Added `--max-opponent-regression` to
  `scripts/seed_matrix_gate.py` so a candidate can fail even if its global
  average looks good but one matched historical opponent regresses too far.
- Refreshed gate report with per-opponent regression enabled:
  `/root/testo-seed-rl/runs/seed_retention_gate_20260531_1530/gate_report.md`.
  No previous repair candidate passed; `3903168` was closest by average but
  failed on `600040` by a -894.5 bb/100 baseline-relative regression.
- Added `scripts/evaluate_seed_candidates.py` as the acceptance workflow:
  evaluate one or more checkpoints with row-level fail-fast gates, write each
  matrix JSONL, then run `scripts/seed_matrix_gate.py` and produce
  `gate_report.md` plus `candidate_status.json`.  It also supports
  `--opponent-file` to reuse the opponent paths from an existing matrix JSONL.
- Added optional training-side checkpoint validation hooks:
  `--checkpoint-validation-command`, `--stop-on-validation-failure`, and
  `--validation-start-hands`.  The trainer injects `SEED_RL_CHECKPOINT`,
  `SEED_RL_CHECKPOINT_DIR`, `SEED_RL_TOTAL_HANDS`, and
  `SEED_RL_METRICS_OUTPUT`, writes `checkpoint_validation` metrics, and can
  stop a run when the external gate fails.
- Added anchor-policy KL regularization for non-sampling-only repair training.
  `--anchor-policy-path` loads a frozen reference actor and `--anchor-kl-coef`
  adds an anchor KL term to PPO actor loss.  This is intended for repair stages
  from `3803147`, so hard-opponent learning is constrained against drifting too
  far from the current best policy.  PPO metrics now include `anchor_kl` and
  `anchor_kl_coef`.
- DeepSeek reviewed anchor-KL repair and suggested a cautious starting coef
  range of `0.01-0.10`, targeting `anchor_kl` roughly `0.1-0.5`.  Added
  `--max-anchor-kl` and `--anchor-kl-patience` so training stops if policy drift
  remains too high across consecutive PPO updates.
- Anchor-KL 50k smoke from `3803147`:
  `/root/testo-seed-rl/runs/seed_anchor_kl_smoke_50k_from3803k_20260531_151954`.
  Training mechanics were healthy: median interval speed about 1,746 hand/s,
  KL median 0.0766, max 0.1190, all-in about 9%, and anchor KL stayed very low
  around 0.0007-0.003.  However the validation gate failed fast:
  `/root/testo-seed-rl/runs/seed_anchor_kl_smoke_eval_3853151/gate_report.md`
  stopped at 12/13 opponents because `seed_rl_hands_900022` scored
  -1371.3 bb/100, a -1838.6 bb/100 regression versus baseline.  Do not use
  `seed_rl_hands_3853151` as a base.
- Fix after smoke: when validation or anchor guard stops a run, skip remaining
  PPO buffer training so the model cannot mutate after the failing checkpoint
  has already been validated.
- Anchor-KL targeted 20k smoke against `900022`:
  `/root/testo-seed-rl/runs/seed_anchor_kl_target900022_20k_from3803k_20260531_152543`.
  Setup used anchor coef `0.1`, LR `2e-7`, and a one-row score file that forced
  `900022` as the hard opponent.  Training was stable, with KL max 0.0543,
  all-in about 9.4%, and online eval +412.7 bb/100 against the active opponent.
  Matrix validation failed immediately at 2/13 opponents:
  `/root/testo-seed-rl/runs/seed_anchor_kl_target900022_eval_3823150/gate_report.md`.
  The failure shifted to `seed_rl_hands_1500036` at -969.8 bb/100 with invalid
  0.12%, confirming that single-opponent repair still causes retention collapse
  even with stronger anchor KL.  Do not use `seed_rl_hands_3823150` as a base.
- Anchor-KL fast-rotation 50k smoke from `3803147`:
  `/root/testo-seed-rl/runs/seed_anchor_kl_fastrotate_50k_from3803k_20260531_152847`.
  This used the full baseline score file, anchor coef `0.1`, LR `2e-7`,
  hard-negative ratio `0.65`, scored-only `1.0`, and opponent updates every
  2k hands.  Training mechanics were stable: 5 PPO updates, KL median 0.0577,
  KL max 0.0734, all-in 9.28% cumulative, invalid/truncated 0, and interval
  speed median 1,637 hand/s.  The safety validation still failed fast:
  `/root/testo-seed-rl/runs/seed_anchor_kl_fastrotate_eval_3853147/gate_report.md`.
  Only 3/13 opponents were evaluated before `seed_rl_hands_500016` dropped to
  -786.6 bb/100, so `seed_rl_hands_3853147` must not replace the baseline.
- Conclusion from the three anchor repair attempts: KL/anchor-KL stability is
  not enough.  A single global opponent model, even when rotated quickly,
  encourages local adaptation and shifts the leak to another historical
  checkpoint.  Added configurable multi-opponent training with
  `--opponent-ensemble-size`; when set above 1, different PPO opponent seats
  route to different loaded historical checkpoints in the same table instead
  of all sharing one global opponent.  The active ensemble paths are logged in
  `metrics.jsonl` for retention analysis.
- DeepSeek review after the fast-rotation failure agreed that the next attempt
  should emphasize hard historical opponents and short smoke validation.  Useful
  takeaways: keep the worst failing opponents heavily represented, run a small
  defensive preflight before broadening the pool, and continue fail-fast matrix
  validation instead of accepting online eval gains.
- First 5k ensemble code smoke:
  `/root/testo-seed-rl/runs/seed_ensemble_smoke_5k_from3803k_20260531_154002`.
  It confirmed `--opponent-ensemble-size 3` works on the server: the trainer
  loaded three simultaneous opponent slots (`500016`, `1000262`, `800200`) and
  later rotated them to another three-checkpoint mix.  Because 5k is shorter
  than the normal PPO buffer, it trained only the final partial buffer and KL
  jumped to 0.1918 despite tiny anchor KL.  Added
  `--skip-final-buffer-train` so short smoke/gated runs can discard the tail
  buffer instead of applying a high-variance final update.
- 50k ensemble smoke:
  `/root/testo-seed-rl/runs/seed_ensemble_smoke_50k_from3803k_20260531_154314`,
  candidate `seed_rl_hands_3853190`.  Training was stable and faster than
  expected for three opponent models: 5 PPO updates, 1,179 hand/s overall,
  median interval speed 1,441 hand/s, KL median 0.0718, KL max 0.0928, all-in
  8.91%, invalid/truncated 0.  Gate report:
  `/root/testo-seed-rl/runs/seed_ensemble_smoke_50k_from3803k_20260531_154314_eval/gate_report.md`.
  It evaluated all 13 opponents but failed: avg +38.6 bb/100 and worst
  `seed_rl_hands_600040` -527.9 bb/100.  This is materially better than the
  single-opponent repairs, but still not acceptable.
- Per-opponent deltas versus `3803147` explain the failure.  The ensemble run
  repaired the old weaknesses (`1000262` from -360.1 to +391.2, `900236` from
  -110.4 to +668.9, `900022` from +467.2 to +545.6) but regressed old strengths
  (`550046` from +420.0 to -273.4, `700016` from +355.7 to -296.4, `600049`
  from +134.4 to -445.4).  Added `--hard-opponent-positive-ratio` so future
  ensemble runs can reserve part of the opponent sampling budget for nonnegative
  baseline matchups instead of over-focusing on negative-score opponents.
- DeepSeek review after the 50k ensemble result recommended exactly this
  direction: force degraded old-strength opponents back into training, keep
  checkpoint rollback/branching, and avoid accepting any candidate until the
  full matrix recovers both hard-negative and retention opponents.
- Balanced positive-retention 50k smoke:
  `/root/testo-seed-rl/runs/seed_ensemble_balanced_50k_from3803k_20260531_154929`,
  candidate `seed_rl_hands_3853170`.  Setup changed only the sampling mix:
  hard-negative ratio `0.45`, positive-retention ratio `0.35`, ensemble size 3,
  LR `1e-7`, anchor coef `0.1`, and skip final buffer enabled.  Training speed
  remained acceptable at 1,134 hand/s, KL median 0.0676, KL max 0.0988, all-in
  9.04%, invalid/truncated 0.  Gate report:
  `/root/testo-seed-rl/runs/seed_ensemble_balanced_50k_from3803k_20260531_154929_eval/gate_report.md`.
  This run failed badly after only 2/13 opponents: `1500036` -1103.3 bb/100
  and `1000054` -555.6 bb/100.  Do not use `3853170`.
- Conclusion: a broad positive-ratio knob is too blunt.  It can destabilize
  small-positive opponents while still failing retention.  The best signal so
  far is the first ensemble 50k run (`3853190`): complete matrix, no severe
  fail-fast collapse, but avg/worst below baseline.  Next mechanism should use
  a fixed multi-bucket roster or candidate selection across multiple short
  checkpoints, not another blind global sampling-ratio tweak.

## Next Work Queue

- Do not start another sampling-only repair from `3803147` unless the proposed
  change includes a different training mechanism.  The next candidate mechanism
  is anchor-KL plus multi-opponent ensemble training, not one global opponent
  rotation.  Do not continue blind sampling-ratio sweeps.  Next suggested
  implementation: a fixed multi-bucket opponent roster that always includes
  hard negatives (`1000262`, `900236`, `800200`), recent regressions
  (`600040`, `600049`, `500016`), and old strengths (`550046`, `700016`,
  `1000054`, `1500036`), plus evaluation of all intermediate checkpoints
  before choosing a continuation base.  Every candidate must pass
  `scripts/seed_matrix_gate.py` before replacing the current best.
- Current best research candidate: `seed_rl_hands_3803147`, but it is not
  acceptable for delivery until worst matchup improves toward -200 bb/100.
- Periodically summarize metrics into this file.
- Add richer all-in diagnostics: all-in by street/SPR, all-in called rate, and
  all-in showdown result.
- Add evaluation stability gates before accepting `seed_rl_best`.
- Reduce KL tail risk with PPO clipping/entropy/buffer tuning if instability
  repeats.
