# Evaluation reliability of IoT botnet / intrusion datasets — code and result tables

This repository contains the analysis code and the raw result tables (CSV/JSON) and figures of an
audit of how evaluation protocols interact with dataset properties on N-BaIoT and seven other
network-traffic datasets, together with an evaluation-reliability profile.

> The accompanying manuscript is under review. Comments, docstrings and printed messages are in English;
> identifiers, file names and table column names are Turkish so that the released tables match the code
> exactly. [GLOSSARY.md](GLOSSARY.md) translates them.

## Manuscript ↔ code

| Manuscript | Code | Result tables |
|---|---|---|
| Sect. 4 (audit: uniqueness, timestamp, shared captures) | `veri/nbaiot_denetim.py`, `veri/nbaiot_jitter_zaman.py`, `veri/nbaiot_zd_birlesim.py`, `faz6_dogrulama.py` (parts A, B) | `faz6_ciktilar/tables/A*.csv`, `B*.csv` |
| Sect. 4.2 (Kitsune initialisation) | `faz7_L_kitsune.py` (uses `kaynak_kod/Kitsune-py`) | `faz7_ciktilar/tables/L_kitsune.csv` |
| Sect. 4.3 (device identifiability) | `faz7_K_kisayol.py` | `faz7_ciktilar/tables/K_kisayol.csv` |
| Sect. 5 (protocols, leakage, seven models) | `faz6_dogrulama.py` (part D), `faz7_Q_protokoller.py`, `faz7_J_gpu_protokoller.py` | `faz6_ciktilar/tables/D_protokoller.csv`, `faz7_ciktilar/tables/QJ_protokoller.csv` |
| Sect. 5.2 (memorisation, near-duplicates) | `faz7_I_ezber.py` | `faz7_ciktilar/tables/I_ezber.csv` |
| Sect. 5.5 (reproduction of the block-disjoint benchmark) | `faz7_H_arumugam.py` (needs the authors' public code) | `faz7_ciktilar/tables/H_*.csv` |
| Sect. 6 (one-class) | `IoT_Faz2–4*.ipynb`, `faz6_E_tek_sinif.py`, `faz7_P_meidan.py` | `faz6_ciktilar/tables/E_tek_sinif.csv`, `faz7_ciktilar/tables/P_meidan.csv` |
| Sect. 7 (reliability profile) | `faz8_metrik.py`, `faz8_analiz.py` | `faz8_ciktilar/tables/faz8_*.csv`, `FAZ8_OZET.md` |
| Sect. 7.4 (ε selection, TOPSIS) | `faz8_eps.py`, `faz8_eps_secim.py` | `faz8_ciktilar/tables/faz8_eps_*.csv`, `FAZ8_EPS.md` |
| Sect. 7.5 (perturbation) | `faz8_bozma.py` | `faz8_ciktilar/tables/faz8c_bozma.csv`, `FAZ8_BOZMA.md` |
| Figures and tables | `makale/sekiller.py`, `makale/tablolar.py` | `makale/figures/`, `makale/tablolar/` |

Seeds, hardware and run times are listed in Appendix A of the manuscript.

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
