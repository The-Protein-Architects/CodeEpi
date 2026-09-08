# codeepi.features

Feature extraction used **only by the prediction CLI**.

Training and reproduction consume pre-computed features from `codeepi_pyg_release_v1.0` and
never call anything here.

| module | purpose |
|---|---|
| `pdb_parser.parse_pdb` | biopython PDB scan → per-residue backbone / seq / resnums |
| `surface.compute_surface` | RSA-thresholded surface residue mask |
| `rsa.compute_rsa` | DSSP-based relative solvent-accessible area |
| `esmif_antigen.encode_antigen` | ESM-IF-1 encoding of antigen-only PDB |
| `esmc.load_precomputed` | load a precomputed ESM-C `.pt` (prediction is precomputed-only) |
| `encode_esmc` | standalone ESM-C **API** encoder — run in the separate `codeepi-esmc` env |



Prediction is **precomputed-only** and needs no network access. The ESM-C
API client (`esm`, EvolutionaryScale) lives ONLY in the separate
`codeepi-esmc` env (`environment-esmc.yml`) — it shares the `esm` import
namespace with `fair-esm`.
