"""Union of timestamp-magnitude values (> 1e9) across the HH_jit mean and variance columns, per class.
Faz 6 B1 counts each column separately; here a row counts if it carries this value in at least one HH_jit column (no training).
Output: faz6_ciktilar/tables/B1b_zd_birlesim.csv (m: mean, v: variance, u: union; row percentage)"""
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
print("attack union %", round(100 * (bm | bv)[k].mean(), 2), " benign %", round(100 * (bm | bv)[~k].mean(), 2))
r.to_csv(os.path.expanduser("~/work/IoT_Botnet/faz6_ciktilar/tables/B1b_zd_birlesim.csv"))
