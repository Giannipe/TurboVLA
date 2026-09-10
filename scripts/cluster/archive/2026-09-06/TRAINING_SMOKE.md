# LIBERO training pipeline smoke test

This is a bounded functionality test, not a full training run and not a success
rate reproduction. The official 97.30% local checkpoint baseline is unchanged.

## Corrected EMA audit

The actual public chain is:

```text
experiments/libero/train.py
  -> turbovla.training.train_mixed
     -> turbovla.training.pi05 (patches the base trainer)
        -> turbovla.training.trainer
```

`pi05.py` already provides AdamW betas (0.9, 0.95), eps 1e-8, EMA decay 0.999,
an EMA update after each optimizer step, and an `ema_model_state_dict` at save.
EMA initializes from the trainable weights before the first optimizer step.
The previous cluster documentation wrongly inferred missing EMA by inspecting
only `trainer.py`. No arbitrary EMA default or raw-weight evaluation bypass is
needed. The issue with the official downloadable export's container key is
separate and remains covered by the SHA256-specific loader compatibility patch.

## Narrow resume fix

The upstream optimizer keeps EMA tensors outside its normal optimizer state
dict. `resume_mode=all` previously restored raw model/Adam/scheduler state but
left EMA initialized from the resumed raw weights. We restore the existing
`ema_model_state_dict` and recorded decay after loading the optimizer state.
Missing EMA or incompatible tensor shapes fail explicitly. This changes resumed
training, not fresh training or the EMA update formula. CPU unit tests compare
the next optimizer and EMA update after resume to uninterrupted toy training.

This does **not** make restart bitwise identical to uninterrupted full training:
the upstream trainer does not restore RNG state or the RLDS iterator position.

## Test stages

1. Read a real episode from each of the four downloaded OpenVLA no_noops suites.
   Check episode metadata, instructions, two 256x256 views, 8-D state, 12x7
   actions, masks and finite values through the official data preprocessing.
2. Run the official mixed-suite training entry point for two optimizer steps.
3. Check finite raw/EMA weights, optimizer step counters, scheduler and loader
   selection of EMA. No TurboVLA task-trained checkpoint is used for initialization.
4. Restart with `resume_mode=all`, stop at step three and validate changed weights.
5. Strict-load the resulting EMA policy and execute task 0 once in each of the
   four suites. Even 0/4 successes can be a successful plumbing test.
6. Write `smoke_summary.json` only after every stage succeeds.

DINOv3 ViT-B and BERT start from their downloaded pretrained backbones;
GroundingDINO initializes the interaction/text projection as in the official
recipe. DINOv3 is unfrozen, BERT frozen; policy weights FP32, DINOv3 BF16 autocast.

Explicit smoke-test differences: one A40 instead of four GPUs, accumulation 16
instead of 4 (8 x 16 x 1 = effective batch 128), zero DataLoader subprocesses,
three total updates rather than 80k, frequent checkpoints, four total evaluation
episodes rather than 2000. The original 80k LR schedule and 10k warmup are kept;
the short run does not rescale the learning-rate schedule. Four-GPU DDP is not
validated by this test. Dataset shuffle/mixing defaults are unchanged.

## Launch and artifacts

```bash
sbatch scripts/cluster/train_libero_smoke.sbatch
```

Artifacts are isolated under:

```text
$SCRATCH_FLASH/TurboVLA/training/libero_smoke/job-<jobid>/
  data_check.json
  train_initial.log
  train_resume.log
  checkpoints/smoke_2.pth
  checkpoints/smoke_3.pth
  checkpoint_2_check.json
  checkpoint_3_check.json
  evaluation/{libero_spatial,libero_object,libero_goal,libero_10}.json
  smoke_summary.json
```

Existing run directories are refused, never overwritten. The job requests one
A40, 8 CPUs, 64 GB RAM and a two-hour upper time limit. Completing this test does
not authorize or start a subsequent long training run.

## Verified run

Job `1921745` completed on 2026-09-06 with exit code 0 in 3 minutes 46 seconds.
Artifacts are under `training/libero_smoke/job-1921745` in the store above.
Four EMA unit tests passed; sample validation passed for all four suites
(metadata totals 1693 episodes). Initial training reached step 2, full-state
resume restored 472 EMA tensors, and training reached step 3 with finite losses.
Both checkpoints passed raw/EMA finiteness, optimizer counter and scheduler
checks. Evaluation strictly loaded 672 EMA tensors and completed four episodes.
The outcome was 0/4 successes after only three optimizer updates: that is not a
benchmark estimate, and the pipeline checks passed independently of success.

Next validation before a full run: four-GPU DDP with per-device batch 8 and
accumulation 4. This smoke test alone does not validate distributed training or
long-run convergence. Existing official checkpoints/results were not changed.
