# CodeEpi

CodeEpi is an ** antigen-only B-cell epitope prediction framework** guided by an antibody-context codebook distilled from antibody–antigen complexes.It ships:

* a trained **teacher** over antibody–antigen complexes,
* a  **4-bin codebook** distilled from the teacher's antigen residue representations,
* an antigen-only **student** distributed as a **5-seed ensemble** —
  inference, evaluation and reproduction all default to the ensemble
  path.

---

## Quickstart 

CodeEpi supports three main usage paths:

1. predict epitopes on a new PDB;
2. reproduce the released 5-seed student results;
3. retrain the full framework.

Choose the matching path below and follow it.

### Common setup

Clone the repository and install the environment:

```bash
git clone https://github.com/<your-org>/CodeEpi.git
cd CodeEpi

conda env create -f environment.yml
conda activate codeepi
```

**ESM-IF weights (manual, one-time).** Prediction needs the ESM-IF-1
checkpoint but **no downloads it automatically** — a prediction run aborts
with these instructions if it is missing. Fetch it once into the torch hub cache:

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
wget -O ~/.cache/torch/hub/checkpoints/esm_if1_gvp4_t16_142M_UR50.pt \
    https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt
```

Using a custom cache location: Export `TORCH_HOME` before running. ESM-C is
handled separately and is **precomputed only** — see
[§1 Predict](#1-predict-on-a-new-pdb) for the one-time `codeepi-esmc` encode step.

Copy the path configuration template:

```bash
cp configs/paths_template.yaml configs/paths.yaml
```

The wrapper scripts use `configs/paths.yaml` when it exists and otherwise
fall back to `configs/paths_template.yaml`.

The default configuration expects the dataset under
`data/codeepi_pyg_release_v1.0/` and the released checkpoints under
`checkpoints_CodeEpi_v1.0/`. Edit `configs/paths.yaml` when storing resources
somewhere else or retraining each stage separately.

### Required external downloads

The preprocessed dataset and released model checkpoints are distributed
separately because they are not stored directly in the GitHub repository.

| Usage path | Dataset | Released checkpoints |
|---|:---:|:---:|
| Predict on a new PDB | No | Yes |
| Reproduce the student results | Yes | Yes |
| Retrain the full framework | Yes | Optional (per stage) |

Download the dataset with:

```bash
bash scripts/download/download_dataset.sh
```

Download the released checkpoints with:

```bash
bash scripts/download/download_checkpoints.sh
```
Full command details: [`docs/release_artifacts.md`](docs/release_artifacts.md)

---

## Three usage paths

### 1. Predict on a new PDB


```bash
cp configs/paths_template.yaml configs/paths.yaml
```

#### 1.1 Antigen-only 5-seed ensemble

Prediction uses a **precomputed** ESM-C tensor only.

**Step 1 — encode the antigen(in the separate `codeepi-esmc` env):**
```bash
# run from the repo root
conda env create -f environment-esmc.yml   
conda activate codeepi-esmc
export ESMC_API_KEY="your_key_here"
python -m codeepi.features.encode_esmc \
    --ag_fasta my_antigen.fasta \
    --antigen_chain A \
    --out my_antigen_esmc.pt
```
The command prints the saved `.pt` path. It encodes the **full-length antigen
SEQRES** (per-chain FASTA records concatenated in `--antigen_chain` order).

**Step 2 — predict (back in the `codeepi` env), feeding that `.pt`:**
```bash
# Optional: customize the epitope threshold** (default 0.45)
conda activate codeepi
ESMC_AG_PT=my_antigen_esmc.pt EPITOPE_THRESHOLD=0.45\
bash scripts/predict/predict_student_antigen.sh my_antigen.pdb my_antigen.fasta A
```
If `ESMC_AG_PT` is missing the run aborts with the exact encode command to run.


**Outputs(single antigen):**

```
outputs/predict/student_antigen/
├── residue_predictions.csv         (includes is_epitope)
├── residue_predictions.json        (includes is_epitope)
├── antigen_colored_by_probability.pdb
├── per_seed_probabilities.csv
├── seqmap.json
└── config_used.yaml
```


#### 1.2 Batch prediction (many antigens from a CSV manifest):**

Each row supplies its own precomputed ESM-C tensor via the manifest's
**required** `esmc_ag_pt` column. Encode every antigen first (Step 1 above,
once per antigen) in the `codeepi-esmc` env, then run the batch in `codeepi`:
```bash
# Optional: customize the epitope threshold** (default 0.45)
conda activate codeepi
EPITOPE_THRESHOLD=0.6
bash scripts/predict/predict_student_antigen.sh --manifest antigens.csv
```

**Outputs(batch):**

```
outputs/predict/student_antigen_batch/
├── <name>/                         (one per manifest row)
│   ├── residue_predictions.csv     (includes is_epitope)
│   ├── residue_predictions.json    (includes is_epitope)
│   ├── antigen_colored_by_probability.pdb
│   ├── seqmap.json
│   ├── per_seed_probabilities.csv
│   └── config_used.yaml
└── batch_summary.json
```

Full command details: [`docs/prediction.md`](docs/prediction.md)

### 2. Reproduce the released 5-seed student results

First fetch the dataset and the released checkpoint bundle:

```bash
bash scripts/download/download_dataset.sh
bash scripts/download/download_checkpoints.sh   # checkpoints for ensemble eval
```

**2.1 evaluate the released checkpoints** The script 
reads the five released seeds from `checkpoints_CodeEpi_v1.0/seeds/`:

```bash
bash scripts/evaluate/evaluate_student_5seed.sh
```
The evaluation result is written to:
results/reproduce/
├── student_5seed_ensemble_metrics_<UTC_TIMESTAMP>.json
└── student_summary.json

**2.2 retrain the 5 student seeds, then evaluate.** Run
locally, or on Slurm as a 5-task array:

```bash
bash scripts/reproduce/reproduce_student_5seed.sh  
```

Each seed writes `student_seed<i>.pt` and `student_seed<i>.json` under
`outputs/reproduce/seed_<i>/`. Evaluate the retrained ensemble by:

```bash
bash scripts/evaluate/evaluate_student_5seed.sh \
    outputs/reproduce/seed_1/student_seed1.pt \
    outputs/reproduce/seed_2/student_seed2.pt \
    outputs/reproduce/seed_3/student_seed3.pt \
    outputs/reproduce/seed_4/student_seed4.pt \
    outputs/reproduce/seed_5/student_seed5.pt
```
The ensemble evaluation result is written to:
results/reproduce/
└── student_5seed_ensemble_metrics_<UTC_TIMESTAMP>.json


reference results are available in:
[`results/reproduce/student_summary.json`](results/reproduce/student_summary.json).

Full command details: [`docs/reproduction.md`](docs/reproduction.md)

### 3. Training the full framework

**3.1 Full retrain: teacher → codebook → 5-seed student.** The generated
codebook is written under `outputs/` and passed straight to the student,
so on this path no `paths.yaml` edit is required:

```bash
bash scripts/download/download_dataset.sh
bash scripts/train/train_all_pipeline.sh
bash scripts/evaluate/evaluate_student_5seed.sh
    outputs/seed_1/student_seed1.pt \                                                             
    outputs/seed_2/student_seed2.pt \                                                               
    outputs/seed_3/student_seed3.pt \                                                                
    outputs/seed_4/student_seed4.pt \                                                                
    outputs/seed_5/student_seed5.pt        

```

Internally: `train_teacher.sh` → auto-exports
`teacher_residue_table.pt` → `build_codebook.sh` → 5 seeds of
`train_student.sh`.

**3.2 Train each stage separately.** 

The connection between stages is a
file on disk. Make sure `configs/paths.yaml` contains
`codebook_checkpoint: outputs/codebook_4bin.pt`:

```bash
# Stage 1: train the teacher (writes best.pt + teacher_residue_table.pt)
bash scripts/train/train_teacher.sh

# Stage 2: build the teacher-derived codebook <seed> default 7876
bash scripts/train/build_codebook.sh \
    outputs/teacher_seed<seed>/teacher_residue_table.pt \
    outputs/codebook_4bin.pt

# Stage 3: train all five student seeds
for seed in 1 2 3 4 5; do
    bash scripts/train/train_student.sh "${seed}"
done

# Evaluate the 5-seed ensemble
bash scripts/evaluate/evaluate_student_5seed.sh
    outputs/seed_1/student_seed1.pt \
    outputs/seed_2/student_seed2.pt \
    outputs/seed_3/student_seed3.pt \
    outputs/seed_4/student_seed4.pt \
    outputs/seed_5/student_seed5.pt

```

See [`docs/training.md`](docs/training.md) for stage-by-stage inputs and
outputs.

---

## Release artifacts

Two external archives are distributed alongside this repository:

| archive | contents |
|---|---|
| `codeepi_pyg_release_v1.0.tar.gz` |  dataset (top-level: `Complex1723/`, `Antigen1321/`, `dataset/`, `full_README.md`) |
| `checkpoints_CodeEpi_v1.0.tar.gz` | `checkpoints_CodeEpi_v1.0/{teacher.pt, codebook_4bin.pt, seeds/student_seed{1..5}.pt}` |

See [`docs/release_artifacts.md`](docs/release_artifacts.md) for the
exact archive layouts and packaging templates.
### Dataset release

The **CodeEpi PyG Release v1.0 dataset** is distributed as
`codeepi_pyg_release_v1.0.tar.gz` through Zenodo.

The archive contains two preprocessed PyTorch Geometric datasets plus a
`dataset/` Python package that both trainers import:

| dataset | samples | purpose |
|---|---:|---|
| Complex1723 | 1,723 | teacher training |
| Antigen1321 | 1,321 | antigen-only student training and reproduction |

After running the dataset download script described in the Quickstart,
the expected layout is:
```
data/codeepi_pyg_release_v1.0/
├── Complex1723/
├── Antigen1321/
├── dataset/
└── full_README.md
```

See [`docs/dataset_release.md`](docs/dataset_release.md) for details.

---

### Checkpoints

Single archive on GitHub Releases / Zenodo:

* `checkpoints_CodeEpi_v1.0.tar.gz` — the supported bundle:
  `teacher.pt`, `codebook_4bin.pt`, and five `seeds/student_seed{1..5}.pt`.

The supported release bundle is
`checkpoints_CodeEpi_v1.0.tar.gz`, available through GitHub Releases and
Zenodo. It contains the `trained teacher`, `codebook_4bin.pt`, and five `seeds/student_seed{1..5}.pt`:

```
checkpoints_CodeEpi_v1.0/
├── teacher.pt
├── codebook_4bin.pt
└── seeds/
    ├── student_seed1.pt
    ├── student_seed2.pt
    ├── student_seed3.pt
    ├── student_seed4.pt
    └── student_seed5.pt
```

See [`checkpoints/README.md`](checkpoints/README.md).


---

## Results

The source for the numeric summary is
[`results/reproduce/student_summary.json`](results/reproduce/student_summary.json).

**The metric space is the PDB-ATOM (resolved) coordinate
system** — every SEQRES residue that has ATOM records in the antigen-only
PDB. Non-surface resolved residues are pinned to probability 0.

---


## Citation

If you use CodeEpi in your work, please cite the accompanying paper.
