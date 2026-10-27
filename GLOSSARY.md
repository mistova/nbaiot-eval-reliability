# Glossary: Turkish identifiers, file names and column names

Comments, docstrings and printed messages are in English. Identifiers, file names and the column
names of the result tables are kept in Turkish so that the released tables match exactly what the
code produces. Column names are built from the words below joined by `_`
(e.g. `test_egitimde_icerik_%` = share of test rows whose content is in the training set).

## Folders, files and abbreviations

| Turkish | English |
|---|---|
| `faz1` … `faz8` | phase 1 … 8 of the study |
| `fazX_ciktilar/` | outputs of phase X |
| `tables/`, `figures/`, `checkpoint/`, `logs/` | result tables, figures, checkpoints (not released), logs (not released) |
| `veri/` | data (download and audit scripts; raw data not released) |
| `kaynak_kod/` | third-party source code (Kitsune-py submodule) |
| `makale/`, `sekiller.py`, `tablolar.py` | manuscript; figure script; table script |
| `OZET` / `ozet` | summary |
| `zd` | timestamp (*zaman damgası*) |
| `hepsi_115` | all 115 features |
| `zd_yok_110` | 110 features without the five timestamp-carrying `HH_jit_*_mean` columns |
| `HHjit_hepsi_yok_100` | timestamp-free set: 100 features without any of the 15 `HH_jit` columns |
| `yalniz_HHjit_15` | only the 15 `HH_jit` columns |
| `S1`–`S6`, `S1f` … | split protocols and their leak-filtered variants (see manuscript, Sect. 3.3) |
| `P0`–`P4` | protocols of the reliability profile (manuscript, Sect. 7.2) |
| `K1`–`K3` | Mirai capture clusters C1–C3 in the manuscript |
| `D-D`, `G-G`, `C-D`, `R-D` | one-class configurations: device model + device threshold, global + global, class model, random group |
| `nbaiot_temiz` | N-BaIoT, timestamp-free variant |

## Words used in identifiers and column names

| Turkish | English | Turkish | English |
|---|---|---|---|
| aday | candidate | kayit | record / run key |
| agirlik, ağırlıklı | weight, weighted | kayip, kayma | loss, shift |
| aile | family (malware family; dataset family) | kazandi | won |
| akis | flow | kip | mode |
| alt, ust | lower, upper | kolay, zor | easy, hard |
| arasi, ici | between, within | korelasyon | correlation |
| ayrintili | detailed | kosu | run |
| bant, genisligi | band, width | kotu | malicious (label 1) |
| baska, diger | other | kuyruk | queue; tail |
| baskin | dominant | log | logarithmic |
| bedeli | cost | makro | macro (average) |
| benzeri | -like | medyan, med | median |
| bolme | split | metrik | metric |
| boyut | dimension, size | not | note |
| bozma | perturbation | okunan | read |
| bozunma | decay | onisleme | preprocessing |
| buyuk | large | oran, orani | ratio, rate |
| (geri) cagirma | recall | ornek, orneklem | sample |
| carpan | factor | ort, ortalama | mean |
| celiskili | conflicting (same vector, different label) | ortak | shared |
| cihaz | device | ozn, oznitelik | feature |
| cogunluk | majority | paket | packet |
| cok | many / multi-class | pay, payi | share |
| deger | value | pencere | window |
| disarida, disi | held out, outside | prim | premium |
| dogru, dogruluk, dogrulugu | correct, accuracy | protokol | protocol |
| doygunluk | saturation | puan | score |
| duzey | level | rastgele | random |
| egitim, egitimde | training, in training | ref | reference |
| entropi | entropy | saldiri | attack |
| epok | epoch | sans | chance |
| esik, esigi | threshold | satir | row |
| eşit | equal | sayisi | count |
| etiket | label | secilen | selected |
| fark, farki | difference | senaryo | scenario |
| filtre | filter | sinif | class |
| gecer | passes | sisme | inflation |
| gecerli | valid | sizinti | leakage |
| gecikme | lag | son | last / end |
| genel | overall | strateji | strategy |
| genisleme | expansion | sure | duration (seconds) |
| gercek | realised / actual | sutun, sutunda | column, in a column |
| gorev | task | tabaninin | of the floor/baseline |
| gorulmemis | unseen | tahmin | prediction |
| grup | group | tam | exact / full |
| gurultu | noise | tavani | ceiling |
| hassasiyet | precision (numeric) | tek | single |
| hata | error | tekil, tekillik | unique, uniqueness |
| havuz | pool | tekrar | duplicate |
| hedef | target | temiz | clean |
| hucre | cell | tip, tipi | type |
| icerik | content (timestamp columns excluded) | tohum | seed |
| ikili | binary | toplam | total |
| ikiz | twin (identical / near-identical row) | tum | all |
| ilk | first | tur | attack type |
| isaret, isaretlenen | flag, flagged | uyum | match |
| istatistik | statistic | uzaklik | distance |
| iyi (huylu) | benign | varlik | entity (device, capture, day, scenario) |
| kal, kalibrasyon | calibration | vektor | vector |
| kararlilik, kararsizlik | stability, instability | veri | data / dataset |
| kararsiz | unstable | yakalama | detection rate |
| kat | fold | yakin | near |
| | | yanlis (alarm) | false (alarm) |
| | | yapilandirma | configuration |
| | | yayilim | spread |
| | | yildiz | star (as in ε*) |
| | | yinelenen, yinelenme | duplicated, duplication |
| | | yuzde, yuzdeligi | percent, percentile |

## Frequent function names

| Function | Purpose |
|---|---|
| `yukle`, `veri` | load a dataset |
| `hashle` | hash rows |
| `bolmeler`, `bolmeler7`, `bolmeler8` | generate split protocols |
| `arindir`, `arindir7` | remove leaked test rows (leak filter) |
| `ck_oku`, `ck_yaz` | read / write checkpoint |
| `kaydet`, `yaz` | save / write output |
| `tavan` | accuracy ceiling κ |
| `kisayol` | best single-feature (shortcut) score |
| `boz` | apply a perturbation |
