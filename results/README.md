# Results

Reference performance for the CodeEpi full-student model, 5-seed
ensemble.

All numbers are computed by the released 5 seed checkpoints
(`checkpoints_CodeEpi_v1.0/seeds/student_seed{1..5}.pt`) plus the fixed 4-bin codebook
(`checkpoints_CodeEpi_v1.0/codebook_4bin.pt`) and inferenced on the release
**test** split (132 antigens) via
`python -m codeepi.evaluate.student_ensemble`
(wrapped by `scripts/evaluate/evaluate_student_5seed.sh`).


## Metric conventions

Test split: 132 antigens. PDB-ATOM (resolved) coordinate system only:

| coordinate system | test residues |
|---|---:|
| PDB-ATOM (resolved) | **34,209** |

* **Macro (per-antigen)** — compute per antigen, then macro-average.
* **Micro (pooled)** — pool residues across all antigens, then compute
  the metric on the pooled arrays.
* MCC / F1 / BACC / AgIoU are reported at both macro and micro. AUPRC /
  AUROC / pAUROC@0.1 are reported at micro.
* **PDB-ATOM (resolved)** — reports in the PDB-ATOM resolved space (34,209
  residues on test); non-surface resolved residues are predicted to
  probability 0 (deterministic true negatives under the fused-surface
  convention). 

## Ensemble headline

Probability-mean ensemble of the 5 seeds. The macro threshold is swept on
the averaged-val macro-MCC and the micro threshold on the averaged-val
micro-MCC, then both are applied to the averaged test probability
(`thr_val_search_macro = 0.45`, `thr_val_search_micro = 0.53`). These
thresholds are recomputed on every invocation of
`evaluate_student_5seed.sh` and are stable across runs on the same
checkpoints.

| metric        | macro (@0.45) | micro (@0.53) |
|---------------|--------------:|--------------:|
| MCC           |    **0.3746** |    **0.3963** |
| F1            |        0.4315 |        0.4412 |
| BACC          |        0.7112 |        0.7211 |
| AgIoU         |        0.3119 |        0.2831 |
| AUPRC         |            —  |        0.4112 |
| AUROC         |            —  |        0.8712 |
| pAUROC@0.1    |            —  |        0.4305 |

## Per-seed independent reference

Each seed picks its own threshold on its own val, then evaluates its own
test at that threshold. CSV group `per_seed_own_opt`.

| seed_index | seed | macro MCC | macro F1 | macro BACC | macro AgIoU | micro MCC | micro F1 | micro BACC | micro AgIoU | micro AUPRC | micro AUROC | micro pAUROC0.1 |
|----------|---:|--------:|-------:|---------:|----------:|--------:|-------:|---------:|----------:|----------:|----------:|--------------:|
| 1 | 1861 | 0.3640 | 0.4207 | 0.7323 | 0.2938 | 0.3746 | 0.4139 | 0.7451 | 0.2610 | 0.3606 | 0.8584 | 0.3962 |
| 2 | 349 | 0.3602 | 0.4204 | 0.7115 | 0.2991 | 0.3764 | 0.4188 | 0.7359 | 0.2649 | 0.3779 | 0.8548 | 0.4096 |
| 3 | 829 | 0.3593 | 0.4195 | 0.7151 | 0.2984 | 0.3655 | 0.4093 | 0.7292 | 0.2573 | 0.3467 | 0.8403 | 0.3949 |
| 4 | 1361 | 0.3570 | 0.4143 | 0.7237 | 0.2905 | 0.3728 | 0.4181 | 0.7236 | 0.2643 | 0.3704 | 0.8558 | 0.4019 |
| 5 | 661 | 0.3562 | 0.4159 | 0.7027 | 0.2984 | 0.3720 | 0.4179 | 0.7201 | 0.2641 | 0.3632 | 0.8606 | 0.3988 |
| **mean±std** | -- | **0.3593 ± 0.0031** | **0.4182 ± 0.0029** | **0.7171 ± 0.0114** | **0.2960 ± 0.0037** | **0.3723 ± 0.0041** | **0.4156 ± 0.0040** | **0.7308 ± 0.0100** | **0.2623 ± 0.0032** | **0.3638 ± 0.0117** | **0.8540 ± 0.0080** | **0.4003 ± 0.0059** |


## Files

* [`reproduce/student_summary.json`](reproduce/student_summary.json) —
  reference JSON summary: ensemble headline (macro@0.45 / micro@0.53), 
  per-seed-at-ensemble-thr block, per-seed independent block, and metadata. 
  All in PDB-ATOM (resolved) space.
* `reproduce/student_5seed_ensemble_metrics_<UTC>.json` — the raw output
  of `scripts/evaluate/evaluate_student_5seed.sh`. A fresh timestamped
  file is written on every run, and nothing is overwritten.
