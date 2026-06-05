# V0 Baseline Optimization Notes

Current frozen baseline:

- name: `baseline_164dim_v0_1000194`
- checkpoint: `seed_rl_hands_1000194`
- architecture: `164` input dims, `12` actions, hidden dims `2048,2048,1024,512`
- latest large-sample result: `13` opponents x `5000` games, avg `+77.7` bb/100, worst `-68.2` bb/100

This checkpoint is the v0 research/test baseline. Do not overwrite or continue
training it in place. New experiments should write to new run directories and
compare back to this baseline.

## Config

The baseline and default guarded smoke settings are recorded in:

`configs/seed_v0_baseline.json`

The config currently includes:

- frozen checkpoint identity and architecture
- hard-subset and full-pool opponent names
- large-sample evaluation summary
- reference-smoke defaults
- row-level safety gates

`scripts/seed_config_export.py` converts this JSON into shell assignments for
existing bash launchers. Environment variables still override config values, so
temporary experiments can be changed without editing JSON.

Example:

```bash
cd /root/testo-seed-rl
SEED_EXPERIMENT_CONFIG=configs/seed_v0_baseline.json \
  bash scripts/run_seed_reference_smoke.sh
```

For a short dry smoke, override the expensive parts:

```bash
cd /root/testo-seed-rl
SEED_EXPERIMENT_CONFIG=configs/seed_v0_baseline.json \
NUM_HANDS=10000 \
VALIDATION_GAMES=100 \
VALIDATION_REPEATS=1 \
REF_STATES_PER_OPPONENT=500 \
REF_MAX_HANDS_PER_OPPONENT=5000 \
  bash scripts/run_seed_reference_smoke.sh
```

## Evaluation

Use `scripts/run_seed_config_eval.py` to run the configured baseline against
the hard subset or full 13-opponent pool without hand-writing checkpoint paths.

Dry-run the full-pool command:

```bash
cd /root/testo-seed-rl
/root/miniconda3/bin/python scripts/run_seed_config_eval.py \
  --config configs/seed_v0_baseline.json \
  --pool full_pool \
  --games 5000 \
  --repeats 1 \
  --dry-run
```

Run the full-pool gate:

```bash
cd /root/testo-seed-rl
/root/miniconda3/bin/python scripts/run_seed_config_eval.py \
  --config configs/seed_v0_baseline.json \
  --pool full_pool \
  --games 5000 \
  --repeats 1
```

The delivery gate in `configs/seed_v0_baseline.json` currently requires avg
`>= 0` bb/100 and worst `>= -200` bb/100 for the configured pool, with clean
invalid/truncated rates.

## Current Direction

Do not run blind PPO fine-tuning from `1000194`. Both Codex and DeepSeek
observed that even conservative PPO repair can improve some opponents while
damaging others. The near-term optimization target is therefore the experiment
workflow:

- keep v0 immutable
- make opponents, gates, and launch parameters config-driven
- fail candidates on row-level min repeat and row std, not just mean score
- compare every new checkpoint against v0 before promotion

Only after the gate workflow is stable should we try new training mechanisms
such as league/population training, robust objective weighting, or explicit
opponent-aware gradient conflict handling.
