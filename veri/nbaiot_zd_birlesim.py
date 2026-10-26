"""Zaman damgası büyüklüğündeki değerlerin (> 1e9) HH_jit ortalama ve varyans sütunlarındaki birleşimi, sınıf başına.
Faz 6 B1 her sütunu ayrı sayar; burada satır en az bir HH_jit sütununda bu değeri taşıyorsa sayılır (eğitim yok).
Çıktı: faz6_ciktilar/tables/B1b_zd_birlesim.csv (m: ortalama, v: varyans, u: birleşim; satır yüzdesi)"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/work/pylibs"))
sys.path.insert(0, os.path.expanduser("~/work/IoT_Botnet"))

import numpy as np
import pandas as pd

from faz7_ortak import yukle

X, M, ozn = yukle()
mi = [j for j, c in enumerate(ozn) if c.startswith("HH_jit_") and c.endswith("_mean")]
vi = [j for j, c in enumerate(ozn) if c.startswith("HH_jit_") and c.endswith("_variance")]
bm = (np.asarray(X[:, mi]) > 1e9).any(1)
bv = (np.asarray(X[:, vi]) > 1e9).any(1)
d = pd.DataFrame({"s": M.sinif.to_numpy(), "m": bm, "v": bv, "u": bm | bv})
r = d.groupby("s")[["m", "v", "u"]].mean().mul(100).round(2)
print(r.to_string())
k = M.kotu.to_numpy() == 1
print("saldırı birleşim %", round(100 * (bm | bv)[k].mean(), 2), " iyi huylu %", round(100 * (bm | bv)[~k].mean(), 2))
r.to_csv(os.path.expanduser("~/work/IoT_Botnet/faz6_ciktilar/tables/B1b_zd_birlesim.csv"))
