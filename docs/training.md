# Training

Train CodeEpi end-to-end on the release. All entry points accept
`--paths configs/paths.yaml`; the wrapper scripts under `scripts/`
default to `configs/paths.yaml` and fall back to
`configs/paths_template.yaml` if the local paths file has not been
created.(All newly generated teacher checkpoints, codebooks, and student checkpoints are saved under `outputs/`. We recommend setting `codebook_checkpoint` to the default path `outputs/codebook_4bin.pt.`)

Standard order:

1. Train teacher (`train_teacher.sh`)
2. Build codebook (`build_codebook.sh`)
3. Train full student (`train_student.sh`)

If you only want to reproduce the reported full-student numbers using
the released `codebook_4bin.pt`, skip steps 1 and 2 and go straight to
`scripts/reproduce/reproduce_student_5seed.sh` — see
`docs/reproduction.md` or `[Section 2 of the top-level README.md]`.

## 1. Train teacher

```bash
bash scripts/train/train_teacher.sh
# or on Slurm
sbatch scripts/sbatch/train_teacher.sbatch
```

Config: `configs/train/teacher.yaml`. Data: `codeepi_pyg_release_v1.0/Complex1723`.
Output: `outputs/teacher_seed<seed>/ckpts/best.pt` plus per-epoch
`history.json` and `results.json`. `best.pt` is selected by **per-antigen
macro MCC on the PDB-ATOM (resolved) val space**. The val threshold picked 
is then applied to test for reporting. 

At the end of the training run the trainer reloads
`best.pt`, and writes:

```
outputs/teacher_seed<seed>/teacher_residue_table.pt
```

This file contains per-training-residue `teacher_prob` and 256-d post-QKV
embeddings (`after_qkv_emb_256`). It is the intermediate artifact
consumed by the codebook builder in step 2. To skip the export (for
example on a resource-constrained node), pass
`--no_export_residue_table` to the trainer.

Then the codebook builder reads
`outputs/teacher_seed<seed>/teacher_residue_table.pt`.

The teacher's role ends at the codebook: it produces the residue table
that the codebook is distilled from. Prediction is antigen-only via the
student ensemble (see `docs/prediction.md`).

## 2. Build teacher-derived codebook

Write the rebuilt codebook to your own path (e.g. under `outputs/`) so the
released `checkpoints_CodeEpi_v1.0/codebook_4bin.pt` is not overwritten:

```bash
bash scripts/train/build_codebook.sh \
    outputs/teacher_seed<seed>/teacher_residue_table.pt \
    outputs/codebook_4bin.pt
```

The two positional arguments are:

1. path to the `teacher_residue_table.pt` produced by step 1,
2. output path for the codebook.

We recommanded to edit `codebook_checkpoint` in your
`configs/paths.yaml`:

```yaml
codebook_checkpoint: outputs/codebook_4bin.pt
```

(Writing to the canonical `checkpoints_CodeEpi_v1.0/codebook_4bin.pt` 
 overwrites the released bundle — not recommended.)

Environment-variable form is also accepted:

```bash
RESIDUE_TABLE=outputs/teacher_seed<seed>/teacher_residue_table.pt \
OUT_PATH=outputs/codebook_4bin.pt \
bash scripts/train/build_codebook.sh
```

The default cut edges `[0.06, 0.50, 0.87]` are the CodeEpi reference
values. Override via `E1=... E2=... E3=... bash …`; the reported results
all use the canonical edges.

## 3. Train full student (5 seeds)

```bash
# one seed at a time (mainly for debugging)
bash scripts/train/train_student.sh 1
# all five in sequence — this is the supported reproduce path
bash scripts/reproduce/reproduce_student_5seed.sh
# or as a Slurm array
sbatch scripts/sbatch/train_student.sbatch
```

Config: `configs/train/student.yaml`. Data:
`codeepi_pyg_release_v1.0/Antigen1321`. Codebook: `outputs/codebook_4bin.pt`.

Each seed's result is written under `outputs/reproduce/seed_<i>/`.
Aggregate the five checkpoints with
`scripts/evaluate/evaluate_student_5seed.sh` to get the ensemble headline.

## 4. End-to-end pipeline (convenience)

```bash
bash scripts/train/train_all_pipeline.sh
```

This wraps steps 1 → 2 → 3, deriving `teacher_residue_table.pt`'s path
from the teacher tag / seed defaults. If your teacher run used a
non-default tag or seed, set `TEACHER_TAG=... TEACHER_SEED=...` before
invoking, or run the three sub-scripts by hand.

## Python vs. sbatch

Every step is available as both a plain Python invocation
(`python -m codeepi.<mod>.<script>`), a shell wrapper (`scripts/train/*.sh`),
and a Slurm launcher (`scripts/sbatch/*.sbatch`). Slurm resource requests
in the launchers are tuned for the reference `a100` partition — adjust
partition / time / memory to match your cluster.

