# codeepi.codebook

CodeEpi teacher-derived 4-bin codebook.

The codebook is:

* `K = 4`
* `bin_edges = [0.06, 0.50, 0.87]` (three cut edges in `(0, 1)`)
* `bin_score` = per-bin geometric mean of the cut edges (`bin_0` floored at `1e-7`)
* `C_weighted_mean` = row-normalized 256-d prototypes per bin, computed as the teacher-predicted-probability-weighted mean of the teacher's post-QKV antibody-context antigen residue embeddings on the training split.

The student head consumes `bin_score` and uses `C_weighted_mean` to initialize its learnable prototypes, with that same value frozen as the L2 anchor.

## Build

```bash
python -m codeepi.codebook.build_codebook \
  --residue-table teacher_residue_table.pt \
  --out checkpoints_CodeEpi_v1.0/codebook_4bin.pt \
  --e1 0.06 --e2 0.50 --e3 0.87
```

