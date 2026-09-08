# codeepi.teacher
group-aware antibody–antigen teacher. Each training sample is a *rep
antigen group*: one representative complex plus every sibling complex that
shares its antigen SEQRES. Cross-attention runs in the rep's fused-surface
coordinate system, so K distinct antibody contexts for the same antigen
sequence are aggregated into a single per-residue prediction.

Modules:

* `group_dataset.build_group_index` — reads only the release metadata
  (`Complex1723/metadata/*`, `Antigen1321` fused surface / epitope) and
  reconstructs the per-rep meta the trainer consumes.
* `group_dataset.GroupTeacherDataset` + `collate_groups` — per-rep dataset
  and its custom collate function; flattens K members with proper node
  offsets and builds the pair tensors above.
* `model.CodeEpiTeacher` — projected-ESM-C fusion → 2×GCN per branch (512
  → 384 → 256) → K-member antigen mean onto the rep frame →
  size-split antibody pool (K=1 mean, K>=2 query-conditioned
  segment softmax with `score_mlp`) → cross-attention (rep antigen as
  query, group-fused CDR as key/value) → per-residue MLP head → logits.
  `forward(..., return_embeddings=True)` additionally returns the 256-d
  post-cross-attention representation (`after_qkv_emb_256`) consumed by
  the codebook builder. `forward` also accepts a per-complex PyG batch
  (each complex treated as a size-1 group, which reduces cleanly to the
  plain-mean fallback).
* `train_teacher.py` — CLI entry point. After training, reloads
  `ckpts/best.pt` and automatically dumps `teacher_residue_table.pt`
  (per-train-rep-position probability + 256-d embedding, one row per
  rep-fused-surface node) into the run directory. That table is the
  codebook builder's input; the format is bit-compatible with the
  pre release. Disable with `--no_export_residue_table`.

Run:

```bash
python -m codeepi.teacher.train_teacher \
  --config configs/train/teacher.yaml \
  --paths configs/paths.yaml
```
