# TurboVLA cluster setup

These helpers keep the Git checkout in the workspace and all large immutable
assets under `${SCRATCH_FLASH}/TurboVLA`.

## Release status (September 2026)

The original download/evaluation helpers below deliberately retain the July
four-checkpoint release for baseline comparisons. For the September **unified**
checkpoint, use [LIBERO_UNIFIED.md](LIBERO_UNIFIED.md). Do not replace old weights
or results in place.

The development worktree `/home/gpepe/ws/TurboVLA-upstream` is on
`setup/upstream-september`, based on upstream `b29ab14`. The active baseline
worktree remains on `setup/cluster-reproduction` while its rollouts finish.
Do not reinstall the shared editable Conda package or change its source during
those runs. Worktree-specific tests must explicitly set their import paths.

Upstream now recommends training batch 128 (8 per device x 4 devices x 4
accumulation steps), not the paper's 256. It describes the new export as EMA at
34k steps; the public trainer does not currently implement/save EMA. The exact
EMA update/decay recipe remains a training-reproduction prerequisite, not an
assumed local default. No training job is launched by these helpers.

Validate the updated worktree (unit tests + CPU strict checkpoint load, not a
second benchmark run):

```bash
TURBOVLA_REPO=/home/gpepe/ws/TurboVLA-upstream \
  sbatch scripts/cluster/validate_upstream_libero.sbatch
```

The updated branch adds a SHA256-allowlisted compatibility path for the official
September export. Other checkpoints still require `ema_model_state_dict`, as
upstream specifies. It prefers actual EMA state when present and does not
silently treat arbitrary raw training checkpoints as EMA.

## 1. Create the policy environments

The two environment names and Python version match the upstream README:

```bash
cd /home/gpepe/ws/TurboVLA
bash scripts/cluster/create_conda_envs.sh all
```

This installs:

- `turbovla-libero`: Python 3.10, PyTorch 2.3.1 + torchvision 0.18.1
  from the CUDA 12.1 wheel index, `mujoco==2.3.7`, and `.[libero]`;
- `turbovla-robotwin`: Python 3.10, PyTorch 2.6.0 + torchvision 0.21.0
  from the CUDA 12.4 wheel index, and `.[robotwin]`.

The RoboTwin pair is fixed explicitly because the project pins torchvision
0.21.0 but leaves torch open-ended. The official matching torch release is
2.6.0. CUDA 12.x wheels are compatible with the newer NVIDIA drivers observed
on this cluster.

FlashAttention is deliberately a separate step. Run this on a node that can
either use the matching binary wheel or compile against a CUDA toolkit:

```bash
bash scripts/cluster/create_conda_envs.sh robotwin --with-flash-attn
```

From the login node, submit the prepared A40 build job instead:

```bash
sbatch scripts/cluster/install_flash_attn.sbatch
```

The RoboTwin simulator itself is a third, separate environment and is not
created by this helper. This script creates the two TurboVLA policy/training
environments requested by the upstream guide.

After a successful setup, preserve the exact resolved dependency graph:

```bash
bash scripts/cluster/capture_env_manifests.sh
```

This writes both `conda list --explicit` and `pip freeze --all` snapshots under
`$SCRATCH_FLASH/TurboVLA/manifests/environments/`.

### Install the external LIBERO simulator

TurboVLA's `libero` extra does not contain the simulator. Install the official
checkout, pinned under flash storage, into the existing policy environment:

```bash
bash scripts/cluster/setup_libero_simulator.sh --plan
bash scripts/cluster/setup_libero_simulator.sh
export LIBERO_CONFIG_PATH="$SCRATCH_FLASH/TurboVLA/config/libero"
```

The helper uses `--no-deps` so LIBERO's legacy requirements cannot overwrite
the tested TurboVLA versions. It also adds an untracked package marker required
by modern setuptools; no tracked LIBERO source file is modified.

## 2. Authenticate for gated DINOv3 weights

Accept the terms on both model pages before starting the download:

- <https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m>
- <https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m>

Then authenticate once from either TurboVLA environment:

```bash
conda activate turbovla-libero
hf auth login
hf auth whoami
```

Do not put a Hugging Face token in the repository or shared scratch. The
download helper redirects model caches but leaves `HF_HOME` unchanged, so the
token remains in the normal private user location.

## 3. Inspect and execute the asset plan

The plan command performs no writes and no network access:

```bash
bash scripts/cluster/download_assets.sh all --plan
```

To let Hugging Face report the transfer sizes without downloading:

```bash
bash scripts/cluster/download_assets.sh all --hf-dry-run
```

The current transfer plan is approximately 33.7 GiB: 6.5 GiB of model assets,
13.3 GiB of training datasets, and 13.9 GiB of RoboTwin simulator archives.

To download and verify every model, checkpoint, training dataset, and simulator
asset:

```bash
bash scripts/cluster/download_assets.sh all
```

On this cluster, prefer the persistent CPU batch job so an SSH or Codex session
limit cannot interrupt the transfer:

```bash
cd /home/gpepe/ws/TurboVLA
sbatch scripts/cluster/download_assets.sbatch all
squeue -u "$USER"
```

The Slurm logs are `/home/gpepe/ws/logs/turbovla-assets-<jobid>.out` and `.err`.
A failed or cancelled download is resumed by submitting the same command.

The script is restartable: `hf download --local-dir` reuses files and download
metadata already present. Every model and dataset revision, including both
DINOv3 snapshots, is pinned and written to `manifests/hf_revisions.tsv`.

The resulting layout is:

```text
$SCRATCH_FLASH/TurboVLA/
├── cache/huggingface/{hub,xet}/
├── pretrained/TurboVLA/
│   └── checkpoints/{libero,robotwin}/
├── models/
│   ├── bert-base-uncased/
│   ├── dinov3-vitb16/
│   ├── dinov3-vitl16/
│   └── groundingdino/
├── datasets/
│   ├── libero/
│   │   └── {libero_10,libero_goal,libero_object,libero_spatial}_no_noops/1.0.0/
│   └── robotwin/
│       └── Clean/<task_name>/
├── simulator_assets/robotwin/
│   └── {background_texture,embodiments,objects}.zip
├── simulators/LIBERO/
├── config/libero/config.yaml
└── manifests/{hf_revisions,libero_simulator_revision}.tsv
```

`Clean/<task_name>` is intentional: it is the layout required by the
TurboVLA RoboTwin registry, even though the upstream dataset has task
directories at its repository root.

For evaluation-only preparation, datasets can be deferred:

```bash
bash scripts/cluster/download_assets.sh models
```

For datasets only:

```bash
bash scripts/cluster/download_assets.sh datasets
```

For RoboTwin simulator archives only:

```bash
bash scripts/cluster/download_assets.sh simulators
```

## 4. Runtime paths

Use local paths explicitly so evaluation does not redownload moving Hugging
Face revisions:

```bash
export TURBOVLA_STORE="$SCRATCH_FLASH/TurboVLA"
export LIBERO_CONFIG_PATH="$TURBOVLA_STORE/config/libero"
export ROBOTWIN_DATA_ROOT="$TURBOVLA_STORE/datasets/robotwin"
export BERT_MODEL_PATH="$TURBOVLA_STORE/models/bert-base-uncased"
export TURBOVLA_INIT_CKPT="$TURBOVLA_STORE/models/groundingdino/groundingdino_swint_ogc.pth"

# LIBERO uses ViT-B; RoboTwin uses ViT-L.
export LIBERO_DINOV3_PATH="$TURBOVLA_STORE/models/dinov3-vitb16"
export DINOV3_MODEL_PATH="$TURBOVLA_STORE/models/dinov3-vitl16"
```

This workspace also has a local modification to
`scripts/robotwin/run_policy_server.sh` that applies local BERT and DINO paths
as checkpoint config overrides. That modification is outside `scripts/cluster`
and is not part of this cluster-only publication; do not assume the unmodified
upstream server automatically consumes these overrides.

The original LIBERO checkpoint names are `object.pth`, `goal.pth`, `spatial.pth`,
and `long.pth`; the upstream `libero_object.pth` example is a documentation
typo.

## 5. Remaining GPU-only setup

- Install RoboTwin policy `flash-attn` from a GPU/build allocation with a
  compatible CUDA toolkit using the command in section 1.
- RoboTwin simulation/evaluation needs a third Python 3.10 environment, Vulkan,
  the official simulator assets, and a compatible simulator checkout. TurboVLA
  requires `script/eval_policy.py`; current RoboTwin `main` removed it, so the
  compatibility checkout must be pinned to
  `c3ddfa8b97d5519efa828b075999bd0006778e5e`. Do not install that simulator
  stack into `turbovla-robotwin`.

## 6. Evaluate the original four-checkpoint LIBERO release

Submit four independent Slurm jobs, one for each official suite/checkpoint
pair. The helper gives all four jobs one shared result directory:

```bash
cd /home/gpepe/ws/TurboVLA
bash scripts/cluster/submit_libero_official.sh
```

The four submissions map to the released checkpoints as follows:

```text
libero_spatial  spatial.pth
libero_object   object.pth
libero_goal     goal.pth
libero_10       long.pth
```

Each job runs all 10 tasks with 50 trials per task (500 episodes), seed 7,
12-step chunks, 12 open-loop actions, 256px observations, and BF16 inference.
Videos are disabled, matching the upstream command and avoiding a large output
tree. Results are written under:

```text
$SCRATCH_FLASH/TurboVLA/results/libero_official_eval/<run-id>/
```

To submit only one suite manually:

```bash
sbatch scripts/cluster/evaluate_libero_official.sbatch libero_object
```

The cluster default is EGL rendering. Override with
`LIBERO_RENDERER=osmesa sbatch ...` only where libOSMesa is installed.

After all four independent jobs finish, validate completeness and print the
per-suite and four-suite average success rates:

```bash
python scripts/cluster/summarize_libero_official.py \
  "$SCRATCH_FLASH/TurboVLA/results/libero_official_eval/<run-id>"
```

This refuses partial or non-reference runs: it requires four correctly mapped
checkpoints, 10 tasks and 500 episodes per suite, seed 7, BF16, and 12-step
open-loop execution. It also reports the difference from the paper values:
99.2 Spatial, 99.8 Object, 97.4 Goal, 94.2 Long, and 97.7 average after
rounding.

### Task-isolated reproduction

The released evaluator notes that the reported 500-episode results ran each
task in an isolated process. To reproduce that execution topology while still
requesting only four Slurm allocations, submit:

```bash
cd /home/gpepe/ws/TurboVLA
bash scripts/cluster/submit_libero_official_isolated.sh
```

Each suite job launches 10 sequential Python processes, one for every task,
then uses `scripts/libero/aggregate_results.py` to validate and combine the 10
task JSON files. The four suite JSON files are written under:

```text
$SCRATCH_FLASH/TurboVLA/results/libero_official_eval_isolated/<run-id>/
```
