# Controlled evaluation of the September 2026 unified release

The completed run documented here was a checkpoint-only comparison against the
old four-checkpoint baseline. No model, preprocessing, simulator, dataset or
environment changes were made during that run.
The shared batch runner has optional checkpoint/preflight overrides; its default
old-release behavior and evaluation arguments are preserved.

Pinned HF revision: `cb5300544693013164c4bb251a13036002a55c81`.
Checkpoint SHA256: `d031ad7be05a2f5d04afb3194ed26b0cb46083685edee7a5e145078a37d26bab`.
Destination: `$SCRATCH_FLASH/TurboVLA/pretrained/TurboVLA-unified-cb53005`.
Results: `$SCRATCH_FLASH/TurboVLA/results/libero_unified_eval/<run-id>`.
Existing releases, manifests and result directories are not overwritten.

## Loader compatibility

Authors describe the new export as the 34k-step EMA checkpoint. Its archive uses
`model_state_dict`, not `ema_model_state_dict`. Upstream commit `ced2b0c` requires
the latter and rejects this export. The completed run used the baseline loader
at `c7c2ba9`, which reads `model_state_dict` and performs a strict load. After
completion we integrated upstream into `setup/cluster-reproduction`, together
with the tested compatibility patch: prefer `ema_model_state_dict`, otherwise
accept `model_state_dict` only for the exact SHA256 above. Neither implementation
computes EMA or renames/rewrites checkpoint keys. EMA provenance is the authors'
statement, not something inferred from the key name.

Sources:

- https://github.com/H-EmbodVis/TurboVLA/issues/11#issuecomment-5504905902
- https://github.com/H-EmbodVis/TurboVLA/issues/12#issuecomment-5504917736
- https://github.com/H-EmbodVis/TurboVLA/commit/ced2b0c465c201ac0c6d41a6855b0c0040360df6
- https://huggingface.co/H-EmbodVis/TurboVLA/tree/cb5300544693013164c4bb251a13036002a55c81

## Launch

```bash
bash scripts/cluster/submit_libero_unified.sh
```

Submits one CPU download/verification job and four independent one-A40 jobs,
dependent on successful download verification (no Slurm arrays). A failed
dependency cancels downstream jobs. Each GPU job verifies checkpoint hash,
statistics and Transformers 4.56.0, then does a strict-load dry run before its
500-episode evaluation. Keep source files unchanged until all jobs finish.

Parameters remain seed 7, 50 trials/task, wait 10, chunk/open-loop 12,
256-pixel observations, relative control, BF16, EGL and offline model loading.
The evaluator runs twice per GPU job: preflight and actual evaluation, in separate
processes. A preflight failure prevents that suite's rollout.

```bash
python scripts/cluster/summarize_libero_official.py /path/to/run --unified-checkpoint
```

The updated upstream training recommendation (batch 128 / EMA) is not implemented
by this experiment: evaluating a downloaded checkpoint requires no retraining.

## Submitted run

Run ID: `20260905T123233Z`.
Download job: `1920961`.
Evaluation jobs: Spatial `1920962`, Object `1920963`, Goal `1920964`, Long `1920965`.
Submission accepted by Slurm; initial status is CPU job pending for priority and
all GPU jobs pending on its successful completion. Logs appear when jobs start,
under `/home/gpepe/ws/logs/<job-name>_<job-id>.out` and `.err`.

Object/Goal/Long initially failed with Slurm `JobLaunchFailure` on
`compute-4-13`, before their scripts started. They were resubmitted in the same
run directory excluding that node: Object `1920976`, Goal `1920977`, Long
`1920978`. Spatial `1920962` was not resubmitted. The cluster-only Git commit
does not change model/evaluator code used by these runs.

All four evaluations finished successfully, with 500 episodes per suite and
1946/2000 successes overall. See [LIBERO_UNIFIED_RESULTS.md](LIBERO_UNIFIED_RESULTS.md).
