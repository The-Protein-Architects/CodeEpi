# Prediction

CodeEpi exposes one prediction CLI. It consumes the single release
bundle `checkpoints_CodeEpi_v1.0.tar.gz` (see
`docs/release_artifacts.md`):

* **`codeepi.predict.student_antigen_ensemble`** — antigen-only 5-seed
  ensemble. Runs on any PDB that has the antigen chain alone. Handles a
  single antigen (`--pdb`;see [Antigen-only 5-seed ensemble]) 
  or a batch of them from a CSV manifest
  (`--manifest`; see [Batch prediction](#batch-prediction-many-antigens)).

The CLI writes this output layout into its `--output` directory (in batch
mode, into a `<name>/` subdirectory per manifest row):

```
<output>/
├── residue_predictions.csv        per-residue probability + bin + rank + is_epitope
├── residue_predictions.json       same, as JSON
├── antigen_colored_by_probability.pdb   PDB with prob*100 in the b-factor column
├── seqmap.json                    SEQRES<->structure anchor mapping + validation
│                                   (seqmap.after_override.json on a --seqmap_override run)
├── per_seed_probabilities.csv     per-seed probability columns
└── config_used.yaml               exact CLI arguments the run consumed
```

`probability_bin ∈ {0, 1, 2, 3}` uses the cut edges from the codebook file
(defaults `[0.06, 0.50, 0.87]`); if you rebuild `codebook_4bin.pt` with
different edges the column moves with it, keeping the reported bins
consistent with the model's internal readout.

`is_epitope` is a boolean column/field marking residues with `probability >= threshold`.
The threshold defaults to **0.45** and can be customized via `--threshold` (Python CLI)
or `EPITOPE_THRESHOLD` (bash wrapper). 

---

## Prerequisites

1. **Extracted checkpoints.** After
   `bash scripts/download/download_checkpoints.sh`:

   ```
   checkpoints_CodeEpi_v1.0/
   ├── codebook_4bin.pt
   └── seeds/
       └── student_seed{1..5}.pt
   ```

2. **ESM-C tensor (precomputed only).** via `--esmc_ag_pt`
   (single mode) or the manifest's `esmc_ag_pt` column (batch mode). The
   run aborts with instructions if it is missing.

   Encode the tensor in the **separate `codeepi-esmc` environment**:

   ```bash
   conda env create -f environment-esmc.yml     
   conda activate codeepi-esmc
   export ESMC_API_KEY="your_key_here"
   python -m codeepi.features.encode_esmc \
       --ag_fasta my_antigen.fasta --antigen_chain A --out my_antigen_esmc.pt
   ```

   The encoder prints the saved `.pt` path; feed it back to prediction
   (in the `codeepi` env) via `--esmc_ag_pt` / `ESMC_AG_PT`. 

   **Precomputed tensor shape contract.** The `.pt` must be a
   `(N, 2560)` float tensor where `N` matches the exact sequence the
   encoder used: the full antigen SEQRES (per-chain FASTA records
   concatenated in `--antigen_chain` order). 


---

## 1 Antigen-only 5-seed ensemble

 `ESMC_AG_PT` is **required** — encode it first (Prerequisite 2):

```bash
#Optional: customize the epitope threshold** (default 0.45)
ESMC_AG_PT=path/to/my_antigen_esmc.pt EPITOPE_THRESHOLD=0.45 \
    bash scripts/predict/predict_student_antigen.sh my_antigen.pdb my_antigen.fasta A
```


Direct CLI (equivalent):

```bash
python -m codeepi.predict.student_antigen_ensemble \
    --pdb          my_antigen.pdb \
    --ag_fasta     my_antigen.fasta \
    --antigen_chain A \
    --checkpoints \
        checkpoints_CodeEpi_v1.0/seeds/student_seed1.pt \
        checkpoints_CodeEpi_v1.0/seeds/student_seed2.pt \
        checkpoints_CodeEpi_v1.0/seeds/student_seed3.pt \
        checkpoints_CodeEpi_v1.0/seeds/student_seed4.pt \
        checkpoints_CodeEpi_v1.0/seeds/student_seed5.pt \
    --codebook     checkpoints_CodeEpi_v1.0/codebook_4bin.pt \
    --esmc_ag_pt   path/to/my_antigen_esmc.pt \
    --output       outputs/predict/student_antigen/ \
    --threshold    0.45
```

### Sequence to structure alignment and validation

Before prediction, the CLI must reconcile two residue spaces. The FASTA
provided through `--ag_fasta` defines the full-length SEQRES used to compute
ESM-C, whereas the PDB ATOM records define the resolved structure used by
ESM-IF and DSSP. These two spaces may differ because of unresolved residues,
insertion codes, or non-contiguous residue numbering.

The anchor aligner (`codeepi.features.seqmap_align`) maps residues position in
the PDB structure back to their corresponding SEQRES positions so that ESM-C
can be cropped and aligned row-for-row with ESM-IF. The mapping and its
validation results are written to `seqmap.json` before model inference.

Most structures are aligned automatically. However, automatic alignment may
fail if the PDB contains unusual insertion codes, non-contiguous
residue numbering, or other irregularities. If any residue fails
validation, the run aborts before prediction and reports the failed residues
in `seqmap.json`.


### Fixing a failed alignment(`--seqmap_override`)

When `success` is false, the terminal message and the `failed_residues` list
in `seqmap.json` identify the residues that could not be aligned.

Two solutions are available:

1. **Check the inputs** — check whether `--ag_fasta` and
   `--antigen_chain` are correct. 

2. **Hand-edit the mapping** — copy `seqmap.json` to a new file and use
   `failed_residues` to identify the problematic residues. For each failed
   residue, determine PDB residue positons correct FASTA sequence 0-based position
   and edit the corresponding `seqres_concat_position`.

For multi-chain antigens, the FASTA sequences are concatenated in
`--antigen_chain` order before positions are indexed.

Re-run the same prediction command with one additional argument:

```bash
--seqmap_override path/to/edited.json
```

---

## 2 Batch prediction (many antigens)

The same CLI runs a whole list of antigens from a CSV **manifest**. 
Single- and batch-antigen modes are mutually exclusive: pass **either** `--pdb`
**or** `--manifest`, never both.

Shell wrapper:

```bash
#Optional: customize the epitope threshold** (default 0.45):
EPITOPE_THRESHOLD=0.6 bash scripts/predict/predict_student_antigen.sh --manifest antigens.csv
# -> outputs/predict/student_antigen_batch/            (can override with arg)
#      <name>/residue_predictions.csv  (the full single-antigen layout)
#      batch_summary.json
```


Direct CLI (equivalent):

```bash
python -m codeepi.predict.student_antigen_ensemble \
    --manifest   antigens.csv \
    --checkpoints \
        checkpoints_CodeEpi_v1.0/seeds/student_seed1.pt \
        checkpoints_CodeEpi_v1.0/seeds/student_seed2.pt \
        checkpoints_CodeEpi_v1.0/seeds/student_seed3.pt \
        checkpoints_CodeEpi_v1.0/seeds/student_seed4.pt \
        checkpoints_CodeEpi_v1.0/seeds/student_seed5.pt \
    --codebook   checkpoints_CodeEpi_v1.0/codebook_4bin.pt \
    --output     outputs/predict/student_antigen_batch/ \
    --threshold  0.45
```

### Manifest format

A CSV with a header row. Required columns `pdb`, `ag_fasta`,
`antigen_chain`, `esmc_ag_pt`; optional columns `name`, `seqmap_override`.

```csv
name,pdb,ag_fasta,antigen_chain,esmc_ag_pt,seqmap_override
7xyz,path/7xyz.pdb,path/7xyz.fasta,A,path/7xyz_esmc.pt,
8abc,path/8abc.pdb,path/8abc.fasta,"A,B",path/8abc_esmc.pt,
7def,path/7def.pdb,path/7def.fasta,A,path/7def_esmc.pt,path/7def_seqmap.json
```

* `name` — output subdirectory under `--output`. If blank, defaults to
  the PDB filename stem. Must be unique across rows.
* `antigen_chain` — comma-separated for multi-chain antigens; please **quote**
  it (e.g. `"A,B"`).
* `esmc_ag_pt` — per-row precomputed ESM-C `.pt` (**required**). Encode each
  antigen first with `codeepi.features.encode_esmc` in the `codeepi-esmc`
  env; a blank cell aborts the batch at manifest-parse time.
* `seqmap_override` — per-row hand-edited seqmap JSON (same as the
  single-antigen `--seqmap_override`; see above).

A single antigen failing (bad alignment, missing file, seqmap gate, …)
is logged, recorded as `failed` in `batch_summary.json`, and the batch
continues with the next row. The run only errors out if **every** row
fails.


