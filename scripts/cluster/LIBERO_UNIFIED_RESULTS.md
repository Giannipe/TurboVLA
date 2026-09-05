# Completed unified LIBERO evaluation — 2026-09-05

Run: `20260905T123233Z`.
Results: `$SCRATCH_FLASH/TurboVLA/results/libero_unified_eval/20260905T123233Z`.
Old baseline: `$SCRATCH_FLASH/TurboVLA/results/libero_official_eval/20260904T171157Z`.

| Suite | Old release | Unified release | Paper | Unified minus paper |
| --- | ---: | ---: | ---: | ---: |
| Spatial | 488/500 (97.6%) | 489/500 (97.8%) | 496/500 (99.2%) | -1.4 pp |
| Object | 496/500 (99.2%) | 496/500 (99.2%) | 499/500 (99.8%) | -0.6 pp |
| Goal | 477/500 (95.4%) | 490/500 (98.0%) | 487/500 (97.4%) | +0.6 pp |
| Long | 462/500 (92.4%) | 471/500 (94.2%) | 471/500 (94.2%) | 0.0 pp |
| Overall | 1923/2000 (96.15%) | 1946/2000 (97.30%) | 1953/2000 (97.65%) | -0.35 pp |

The paper prints 97.7% after rounding. The unified release gains 23 successes
over our old-release baseline but is 7 successes below the paper in aggregate.
Long matches the suite-level rate; Goal exceeds it. This is not an exact
reproduction of all four published rates and is not proof of statistical
equivalence or matching per-episode outcomes with the authors.

## Per-task successes (50 trials each; task indices 0 through 9)

| Suite | Old release | Unified release |
| --- | --- | --- |
| Spatial | 50,50,50,50,47,48,50,49,48,46 | 50,50,50,48,47,47,49,49,49,50 |
| Object | 50,50,49,47,50,50,50,50,50,50 | 50,50,50,49,50,48,50,50,50,49 |
| Goal | 50,49,47,43,48,50,48,50,48,44 | 50,49,50,46,50,49,49,50,50,47 |
| Long | 34,49,50,47,47,45,41,50,49,50 | 46,50,49,47,48,46,44,50,45,46 |

Spatial initially looked worse on a partial prefix, but finished one success
above the old release. Only comparisons on matching episodes or complete suites
are meaningful; a partial running rate should not be compared to a full mean.

## Checks and provenance

- All suites have exactly task IDs 0..9 and 50 unique episodes 0..49 per task.
- All 2000 logged episode outcomes match per-task and suite JSON counts.
- All four stdout logs contain the normal finished marker.
- Jobs: Spatial 1920962, Object 1920976, Goal 1920977, Long 1920978.
- The latter three replaced jobs that failed in Slurm before starting Python.
- Same unified checkpoint for every suite: HF revision
  `cb5300544693013164c4bb251a13036002a55c81`, SHA256
  `d031ad7be05a2f5d04afb3194ed26b0cb46083685edee7a5e145078a37d26bab`.
- Model/evaluator source at job startup: `c7c2ba9`; the subsequent cluster-only
  commit did not alter this source. Upstream integration happened after the run.
- Strict checkpoint loading succeeded for 672 tensors.
- Seed 7; 50 trials/task; chunk/open-loop 12; wait 10; 256px; relative control;
  BF16; EGL; A40; torch 2.3.1+cu121; Transformers 4.56.0; MuJoCo 2.3.7;
  robosuite 1.4.1; unchanged release normalization statistics.
- The updated upstream plus hash-guarded loader patch passed five unit tests
  and a complete CPU strict-load check in job 1920979. That was not another
  2000-episode rollout and must not be reported as one.

## Interpretation and next stage

We followed the newly recommended single-checkpoint release, with an explicitly
documented loader compatibility exception. Exact authors' simulator/rendering
stack and original evaluation traces remain unverified. Do not claim that the
remaining discrepancy is caused by a particular GPU or renderer without a
controlled comparison.

Training has not started. The updated training recommendation is global batch
128, but the exact EMA training/update recipe still needs clarification: the
public trainer does not currently implement/save EMA. Training loss alone is not
the authors' checkpoint-selection criterion.

Sources:

- https://arxiv.org/html/2607.27205v2#S5
- https://github.com/H-EmbodVis/TurboVLA/issues/11#issuecomment-5504905902
- https://github.com/H-EmbodVis/TurboVLA/issues/12#issuecomment-5504917736
