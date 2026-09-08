# Reproducing CodeEpi reported student results

This page documents how to reproduce the CodeEpi **full student** results
reported in the top-level `README.md` and in
`results/reproduce/student_summary.json`.

## What is reported

The reported student result is the **5-seed ensemble** on the release
`test` split, with a single macro threshold `thr_val_search_macro` swept
on the ensemble-averaged val probability. **PDB-ATOM (resolved) is the
primary metric space with non-surface residues pinned to probability 0. 


## Prerequisites

1. **Dataset release.** A local copy of the CodeEpi-compatible
   `codeepi_pyg_release_v1.0/` directory. Copy
   `configs/paths_template.yaml` to `configs/paths.yaml` and set
   `dataset_root` to point at your local unpack. See
   `docs/dataset_release.md`.
2. **Codebook checkpoint.** `checkpoints_CodeEpi_v1.0/codebook_4bin.pt`. Either from
   the released bundle (`bash scripts/download/download_checkpoints.sh`)
   or one rebuilt via `codeepi.codebook.build_codebook`.
3. **Environment.** `conda env create -f environment.yml`.

## Command

Reproduce all five seeds:

```bash
bash scripts/reproduce/reproduce_student_5seed.sh
```

Or as a Slurm array (one seed per array task):

```bash
sbatch scripts/sbatch/train_student.sbatch
```

Each seed writes:

```
outputs/reproduce/seed_<i>/student_seed<i>.json
outputs/reproduce/seed_<i>/student_seed<i>.pt
```

Ensemble numbers require evaluating the 5 checkpoints together; see the
next section.

## Producing the ensemble summary

Once the 5 seed `.pt` files exist, evaluate them jointly. The evaluation
script accepts the 5 checkpoint paths as arguments, so retrained seeds are
ensembled straight out of `outputs/reproduce/` — no copying, and the
released checkpoints under `checkpoints_CodeEpi_v1.0/seeds/` are never
overwritten:

```bash
bash scripts/evaluate/evaluate_student_5seed.sh \
    outputs/reproduce/seed_1/student_seed1.pt \
    outputs/reproduce/seed_2/student_seed2.pt \
    outputs/reproduce/seed_3/student_seed3.pt \
    outputs/reproduce/seed_4/student_seed4.pt \
    outputs/reproduce/seed_5/student_seed5.pt
```

To instead score the released checkpoints
(`bash scripts/download/download_checkpoints.sh`), run the script with no
arguments and it falls back to `checkpoints_CodeEpi_v1.0/seeds/`:

```bash
bash scripts/evaluate/evaluate_student_5seed.sh
```

Each run writes a fresh, timestamped
`results/reproduce/student_5seed_ensemble_metrics_<UTC>.json` (nothing is
ever overwritten; set the `ENS_OUT` environment variable to force a fixed
path). The file contains:

* `ensemble_prob` block — metrics on the ensemble-averaged test probability at
  `thr_val_search_macro` and `thr_val_search_micro` (in the PDB-ATOM
  resolved space). 
* `per_seed_test_at_ensemble_thr` — each seed's own test probability evaluated at
  the shared ensemble threshold, aggregated to mean±std across seeds.
* `per_seed_independent_eval` — evaluates its own test at own threshold.
Compare your numbers against the reference block in
`results/reproduce/student_summary.json` (which itself was produced by
the same `codeepi.evaluate.student_ensemble` pipeline against the
released checkpoints).

## Notes on determinism

The trainer seeds `random`, `numpy`, and `torch` (CPU + all CUDA devices)
via `codeepi.utils.seed.seed_everything`. cuDNN determinism flags are
intentionally left at PyTorch defaults — flipping them degrades
throughput and was not required to reproduce the reported means. Some
run-to-run variance across metrics is expected even with fixed seeds
when running across different GPU generations.

## Extending to more seeds

Edit `configs/train/student.yaml` to grow the `seeds:` list and re-run
`train_student.sh` for each new `--seed_index`. Then pass whatever subset
of `.pt` paths you want to ensemble directly to
`evaluate_student_5seed.sh` as arguments (the underlying
`codeepi.evaluate.student_ensemble` accepts any number `>= 2`).
