# Checkpoints

This directory is intentionally empty in the GitHub repository. CodeEpi
ships checkpoints as an **external release asset**.

## Bundle: `checkpoints_CodeEpi_v1.0.tar.gz`

The release ships as a single tarball. Extract from the repo root:

```bash
cd /path/to/CodeEpi
tar -xzf checkpoints_CodeEpi_v1.0.tar.gz
```

This places `checkpoints_CodeEpi_v1.0/` directly under the repo root:

```
checkpoints_CodeEpi_v1.0/
├── teacher.pt              # trained CodeEpi teacher over antibody-antigen complexes
├── codebook_4bin.pt        # teacher-derived 4-bin codebook
└── seeds/
    ├── student_seed1.pt   
    ├── student_seed2.pt  
    ├── student_seed3.pt   
    ├── student_seed4.pt  
    └── student_seed5.pt  
```

All scripts and configs reference `checkpoints_CodeEpi_v1.0/` with repo-relative
paths, so no further setup is needed after extraction.

The five seed checkpoints define the release **5-seed ensemble** and are the
ONLY supported inference / evaluation path. All public entry points
(`scripts/predict/predict_student_antigen.sh`,
`scripts/evaluate/evaluate_student_5seed.sh`,
`scripts/reproduce/reproduce_student_5seed.sh`) consume this bundle
directly. 

## Download

```bash
bash scripts/download/download_checkpoints.sh
```

Sets `CODEEPI_CHECKPOINTS_URL` to override the source URL if the release
lives outside the Zenodo default.


Reference numbers live in `results/reproduce/student_summary.json` and
`results/README.md`.

