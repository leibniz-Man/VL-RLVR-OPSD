# Phase 1 Conflict Energy Analysis

This experiment directory is for the token-level conflict-energy diagnostic
described in `phase1_token_level_conflict_energy_experiment.md`.

The directory is intentionally independent from the CEPO launch scripts. The
only training-code hook is a narrow token-record dump inside
`verl/trainer/ray_trainer.py`.

## What Can Run Now

`scripts/analyze_existing_artifacts.py` analyzes artifacts that already exist in
the CEPO checkout:

- GRPO checkpoint tracker and LMMS evaluation result JSON.
- GRPO and CEPO training logs.
- CEPO histogram dumps such as `cepo_histograms/step_0005.json`.

This gives a preliminary mechanism check:

- GRPO baseline checkpoint/eval summary.
- CEPO delta distribution on positive and negative rollout tokens.
- Directional wrong-rate diagnostics from the existing CEPO run.

This is not yet the full token-level conflict-energy test, because the current
saved artifacts do not contain per-token `prompt_id`, `rollout_id`, `token_id`,
`reward`, `advantage`, `pos_teacher_logprob`, and `neg_teacher_logprob` together.

## Full Metric Script

`scripts/token_conflict_energy.py` computes the main Phase 1 metrics from a
token-record JSONL/CSV file. Expected columns:

- `prompt_id`
- `rollout_id`
- `token_position`
- `token_id`
- `token_text`
- `reward`
- `advantage`
- `delta_cepo`

Optional columns:

- `token_weight_cepo`
- `effective_adv_grpo`
- `effective_adv_cepo`

If optional effective advantages are missing, the script computes:

- `effective_adv_grpo = advantage`
- `token_weight_cepo = clip(exp(sign(advantage) * delta_cepo), 1 - eps_w, 1 + eps_w)`
- `effective_adv_cepo = advantage * ((1 - lam) + lam * token_weight_cepo)`

## Training-Time Dump

`verl/trainer/ray_trainer.py` now writes token records inside the CEPO advantage
block only for:

```text
global_step in {1, 5, 10}
```

The dump is written under the active CEPO checkpoint root:

```text
<trainer.save_checkpoint_path>/phase1_token_records/
```

For example:

```text
checkpoints/cepo/qwen3_vl_2b_geo_cepo/phase1_token_records/step_0001.jsonl
checkpoints/cepo/qwen3_vl_2b_geo_cepo/phase1_token_records/step_0001.summary.json
```

Only mixed-outcome prompt groups are dumped, because all-correct and all-wrong
groups cannot contribute to positive-vs-negative conflict energy.

## Suggested Workflow

1. Run the existing-artifact analysis first:

```bash
bash experiments/phase1_conflict_energy/run_existing_artifacts.sh
```

2. Run a short CEPO job. Token records will be emitted automatically at steps
   1, 5, and 10.

3. Run the full metric:

```bash
python experiments/phase1_conflict_energy/scripts/token_conflict_energy.py \
  --input checkpoints/cepo/qwen3_vl_2b_geo_cepo/phase1_token_records/step_0005.jsonl \
  --output-dir experiments/phase1_conflict_energy/outputs/token_conflict
```

The primary table is `group_summary.csv`; the most useful manual-inspection file
is `top_conflicts.csv`.
