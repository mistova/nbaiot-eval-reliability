# Evaluation reliability of IoT botnet / intrusion datasets — code and result tables

This repository contains the analysis code and the raw result tables (CSV/JSON) and figures of an
audit of how evaluation protocols interact with dataset properties on N-BaIoT and seven other
network-traffic datasets, together with an evaluation-reliability profile.

> The accompanying manuscript is in preparation. The written interpretation of the results will be
> added to this repository at submission time. Code comments and table column names are in Turkish.

## Layout

| Path | Content |
|---|---|
| `IoT_Faz1_*.ipynb` … `IoT_Faz5_*.ipynb` | Phases 1–5: data exploration, one-class (autoencoder) regime, mechanism, thresholds, seed stability, split protocols |
| `faz6_*.py`, `faz6_calistir.sh` | Phase 6: verification experiments (float64 audit, feature-group ablations, split protocols, one-class runs) |
| `faz7_*.py`, `faz7_kuyruk*.sh` | Phase 7: reproduction of a block-disjoint protocol, 7 models × 5 seeds, nearest-neighbour baseline, single-feature shortcut, original-protocol replication, Kitsune feature extractor variants, other datasets |
| `faz8_metrik.py`, `faz8_analiz.py` | Phase 8: evaluation-reliability profile on 9 datasets × 5 protocols |
| `faz8_eps.py`, `faz8_eps_secim.py` | Phase 8b: near-duplicate threshold (ε) selection with TOPSIS and nested leave-one-family-out validation |
| `faz8_bozma.py` | Phase 8c: controlled perturbation experiment (noise, quantisation, thinning, feature subsets) |
| `veri/` | Dataset audit scripts, download helper, summary JSON |
| `faz*_ciktilar/tables`, `faz*_ciktilar/figures` | Result tables and figures produced by the scripts above |
| `kaynak_kod/Kitsune-py` | Git submodule pointing to the original Kitsune implementation (unmodified) |

Checkpoints, caches and logs are not included; every table can be regenerated from the scripts.

## Datasets (not redistributed — download from the original sources)

| Dataset | Source |
|---|---|
| N-BaIoT | Meidan, Y., Bohadana, M., et al.: detection_of_IoT_botnet_attacks_N_BaIoT. UCI Machine Learning Repository (2018). https://doi.org/10.24432/C5RC8J |
| Kitsune | Kitsune Network Attack [Dataset]. UCI Machine Learning Repository (2019). https://doi.org/10.24432/C5D90Q |
| IoT-23 | Garcia, S., Parmisano, A., Erquiaga, M.J.: IoT-23. Zenodo (2020). https://doi.org/10.5281/zenodo.4743746 |
| MedBIoT | https://cs.taltech.ee/research/data/medbiot/ |
| CIC-IoT2023 | https://www.unb.ca/cic/datasets/iotdataset-2023.html |
| CICIDS2017 (original) | https://www.unb.ca/cic/datasets/ids-2017.html |
| CICIDS2017 (Engelen et al. 2021 corrected) | https://intrusion-detection.distrinet-research.be/WTMC2021/ |
| Bot-IoT (5 % subset) | https://research.unsw.edu.au/projects/bot-iot-dataset |

`veri/indir_dis_veri.sh` downloads Kitsune and MedBIoT. The scripts expect the data under
`~/work/IoT_Botnet/veri/`; adjust the path constants at the top of each script if needed.

## Environment

Python 3.13; see `requirements.txt`. Phases 7–8 use a CUDA GPU (XGBoost `device="cuda"`, PyTorch
nearest-neighbour search); the largest jobs need ~90 GB RAM. Clone with submodules:

```
git clone --recurse-submodules <this repository>
```

## License

Code: MIT (see `LICENSE`). Datasets remain under their original licences.
