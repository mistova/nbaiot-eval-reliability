"""Phase 7 shared helpers. Uses the Phase 6 data cache (X64.npy, meta.pkl, hashes) and split definitions."""
import os
import pickle
import sys
import time

_PYLIBS = os.path.expanduser("~/work/pylibs")
if _PYLIBS not in sys.path:
    sys.path.insert(0, _PYLIBS)

import numpy as np
import pandas as pd

from faz6_dogrulama import MIRAI_KUME, arindir, bolmeler, satir_hash, yukle, zd_ozn  # noqa: F401

OUT = os.path.expanduser("~/work/IoT_Botnet/faz7_ciktilar")
TAB, CACHE, CK = f"{OUT}/tables", f"{OUT}/cache", f"{OUT}/checkpoint"
for d in (TAB, CACHE, CK, f"{OUT}/logs"):
    os.makedirs(d, exist_ok=True)
UST = 150_000
BLOK = 500
TOHUMLAR = (42, 7, 123, 2024, 13)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ck_oku(ad):
    yol = f"{CK}/{ad}.pkl"
    return pickle.load(open(yol, "rb")) if os.path.exists(yol) else []


def ck_yaz(ad, son):
    yol = f"{CK}/{ad}.pkl"
    pickle.dump(son, open(yol + ".tmp", "wb"))
    os.replace(yol + ".tmp", yol)


def icerik_hash(X, ozn):
    """float64 content hash excluding the 5 HH_jit_*_mean columns that carry timestamps (same file as in Phase 6)."""
    return satir_hash(X, "float64", haric=set(zd_ozn(ozn)), ad="float64_zdsiz")


def blok_id(M, b=BLOK):
    """Consecutive blocks of b rows within each (cihaz, aile, tur) file (the unit used in Arumugam 2026)."""
    dosya = M.groupby(["cihaz", "aile", "tur"], sort=False).ngroup().to_numpy().astype(np.int64)
    return dosya * 100_000 + (M.satir_no.to_numpy() // b)


def bolmeler7(M, H, seed):
    """Phase 6 splits (S1-S5) + S6: random 80/20 split over the whole data, disjoint by 500-row blocks."""
    out = bolmeler(M, H, seed)
    g = blok_id(M)
    ug = np.unique(g)
    test_g = np.random.RandomState(seed).permutation(ug)[: int(0.2 * len(ug))]
    m = np.isin(g, test_g)
    out.append(("S6", "-", np.flatnonzero(~m), np.flatnonzero(m)))
    return out


def arindir7(bl, H):
    """Removes test rows whose content hash occurs in training (S1f, S3f, S4f, S6f)."""
    out = arindir(bl, H)
    for b, k, tr, te in bl:
        if b == "S6":
            out.append(("S6f", k, tr, te[~np.isin(H[te], np.unique(H[tr]))]))
    return out


def oznitelik_setleri(ozn):
    zi = set(zd_ozn(ozn))
    jit = {i for i, c in enumerate(ozn) if c.startswith("HH_jit_")}
    n = len(ozn)
    return {"hepsi_115": np.arange(n),
            "zd_yok_110": np.array([i for i in range(n) if i not in zi]),
            "HHjit_hepsi_yok_100": np.array([i for i in range(n) if i not in jit])}


def isaretli_log(X):
    """Signed log1p: compresses timestamps (~1.5e9) and magnitudes up to 1e21."""
    return np.sign(X) * np.log1p(np.abs(X))
