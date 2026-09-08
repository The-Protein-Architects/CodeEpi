# Release artifacts

CodeEpi ships two external artifacts alongside this repository. Neither
is stored in git; they are distributed via Zenodo / GitHub Releases and
referenced from the download scripts under `scripts/download/`.


## 1. Dataset archive

* Archive: `codeepi_pyg_release_v1.0.tar.gz`
* Downloader: `scripts/download/download_dataset.sh`
* Target env var: `CODEEPI_DATASET_URL`
* detail in: `dataset_release.md`

Layout after extraction:

```
codeepi_pyg_release_v1.0/
├── Complex1723/
├── Antigen1321/
├── dataset/
└── full_README.md
```

## 2. Checkpoint archive

* Archive: `checkpoints_CodeEpi_v1.0.tar.gz`
* Downloader: `scripts/download/download_checkpoints.sh`
* Target env var: `CODEEPI_CHECKPOINTS_URL`

Layout after extraction:

```
checkpoints_CodeEpi_v1.0/
├── teacher.pt              # trained CodeEpi teacher
├── codebook_4bin.pt        # V4cB 4-bin codebook, bin_edges=[0.06, 0.50, 0.87]
└── seeds/
    ├── student_seed1.pt    
    ├── student_seed2.pt    
    ├── student_seed3.pt   
    ├── student_seed4.pt   
    └── student_seed5.pt   
```

