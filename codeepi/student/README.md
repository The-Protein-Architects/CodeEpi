# codeepi/student -- Full student model

This package contains the CodeEpi **antigen-only student**: the model users
run when they only have an antigen structure and want per-residue B-cell
epitope probabilities.

Files
-----

- `model.py` -- `CodeEpiStudent` = `CodeEpiStudentBackbone` (3xEGNN) + `ProtoHead`
  (prototype-attention over a fixed 4-bin teacher-derived codebook). No auxiliary
  head, no local pair-attention path.
- `losses.py` -- asymmetric focal loss on probability (`afl_p`), the L2 anchor
  distance helper on the prototype head, cosine `tau` schedule for the softmax
  temperature, and a parameter-space `EMA` wrapper.
- `train_student.py` -- CLI trainer that consumes
  `configs/train/student.yaml` + `configs/paths.yaml`, reads
  the CodeEpi PyG Release v1.0 via `codeepi.data.release.AntigenReleaseDataset`, and
  writes a paired `.json` / `.pt` per seed.

Student inference does **not** need the full teacher model at runtime. The
student consumes only the antigen structure and the fixed teacher-derived
codebook for guidance (`checkpoints_CodeEpi_v1.0/codebook_4bin.pt`).
