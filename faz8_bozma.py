"""Faz 8c: ε uyarlama rehberi — denetimli bozma deneyi (kullanıcı onayı 2026-10-23).
Her (veri, bozma, düzey) koşulunda satır düzeyinde en iyi ε (ε*):
  şişmiş satır = P0 (rastgele 70/30) modeli doğru, satırın varlığı eğitimden tümüyle çıkarılmış model (P2) yanlış
  d            = test satırının P0 eğitimindeki en yakın komşu mesafesi / √boyut (işaretli log + z-skoru)
  ε*           = şişmiş satırları "d ≤ ε" ile en iyi ayıran eşik (Youden J); 200 bootstrap ile %95 aralık
Bozmalar: göreli gürültü, anlamlı basamağa nicemleme, satır seyreltme (kayan pencere sıklığı), öznitelik alt kümesi.
Çıktı: checkpoint/faz8c_<veri>.pkl, tables/faz8c_bozma.csv"""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import faz8_metrik as M
from faz7_ortak import log
from faz7_gpu_modeller import Onisleme, en_yakin

KOSUL = [("taban", 0), ("gurultu", 0.001), ("gurultu", 0.01), ("gurultu", 0.05), ("nicem", 3), ("nicem", 2),
         ("seyrelt", 4), ("seyrelt", 16), ("boyut", 0.5), ("boyut", 0.25)]
IZGARA = np.logspace(-5, 0.5, 56)
N_BOOT = 200


def boz(X, tur, duzey):
    """(X', satır indeksi, sütun indeksi). Büyük veride bellek için float32 ve parça parça."""
    n, p = X.shape
    if tur == "taban":
        return X, np.arange(n), np.arange(p)
    if tur == "seyrelt":
        s = np.arange(0, n, int(duzey)); return X[s], s, np.arange(p)
    if tur == "boyut":
        c = np.sort(np.random.RandomState(3).choice(p, max(2, int(round(duzey * p))), replace=False))
        return np.ascontiguousarray(X[:, c]), np.arange(n), c
    Y = np.empty((n, p), np.float32)
    rs = np.random.RandomState(5)
    for i in range(0, n, 500_000):
        b = np.asarray(X[i:i + 500_000], dtype=np.float64)
        if tur == "gurultu":
            b = b * (1 + duzey * rs.standard_normal(b.shape))
        elif tur == "nicem":          # anlamlı basamağa yuvarlama
            with np.errstate(divide="ignore", invalid="ignore"):
                k = np.where(b == 0, 0, duzey - 1 - np.floor(np.log10(np.abs(b))))
            b = np.round(b * 10.0 ** k) / 10.0 ** k
            b[~np.isfinite(b)] = 0
        Y[i:i + 500_000] = b
    return Y, np.arange(n), np.arange(p)


def kosul_olc(ad, X, y, v, tur, duzey):
    n = len(y)
    rs = np.random.RandomState(42)
    p = rs.permutation(n); k = int(0.7 * n)
    tr, te = p[:k], p[k:]
    rs0 = np.random.RandomState(0)
    te_s = te if len(te) <= M.N_TE else rs0.choice(te, M.N_TE, replace=False)
    tr_nn = tr if len(tr) <= 2_000_000 else rs0.choice(tr, 2_000_000, replace=False)
    on = Onisleme().fit(np.asarray(X[rs0.choice(tr, min(len(tr), 300_000), replace=False)], dtype=np.float64))
    Z = np.concatenate([on(np.asarray(X[tr_nn[i:i + 500_000]], dtype=np.float64)) for i in range(0, len(tr_nn), 500_000)])
    D, _ = en_yakin(Z, on(np.asarray(X[te_s], dtype=np.float64)), k=1, parca_te=2048, parca_tr=500_000)
    del Z
    d = D[:, 0] / np.sqrt(X.shape[1])
    # P0 modeli
    tr_m = tr if len(tr) <= M.N_TR else np.random.RandomState(42).choice(tr, M.N_TR, replace=False)
    c0 = M.xgb_f1(np.asarray(X[tr_m]), y[tr_m], np.asarray(X[te_s]), y[te_s], 42) == y[te_s]
    # varlık dışarıda referansı: her test satırı, kendi varlığı hiç görülmemiş modelle
    c2 = np.full(len(te_s), np.nan)
    for e in np.unique(v[te_s]):
        tr_e = np.flatnonzero(v != e)
        if len(tr_e) < 100:
            continue
        sec = np.flatnonzero(v[te_s] == e)
        sec = sec[np.isin(y[te_s[sec]], np.unique(y[tr_e]))]      # görülmemiş sınıf dışarıda
        if len(sec) == 0:
            continue
        tr_e = tr_e if len(tr_e) <= M.N_TR else np.random.RandomState(42).choice(tr_e, M.N_TR, replace=False)
        c2[sec] = M.xgb_f1(np.asarray(X[tr_e]), y[tr_e], np.asarray(X[te_s[sec]]), y[te_s[sec]], 42) == y[te_s[sec]]
    ok = ~np.isnan(c2)
    s = (c0 & (c2 == 0))[ok].astype(int); dd = d[ok]
    r = {"veri": ad, "bozma": tur, "duzey": duzey, "n": n, "boyut": X.shape[1], "n_gecerli": int(ok.sum()),
         "dogru_P0": float(c0[ok].mean()), "dogru_P2": float(c2[ok].mean()), "sisme_payi": float(s.mean()),
         "tau0_benzeri": float((dd == 0).mean()), "tau_0.01": float((dd <= 0.01).mean()), "tau_0.1": float((dd <= 0.1).mean()),
         "d_medyan": float(np.median(dd))}
    if s.sum() < 30 or s.sum() > len(s) - 30:
        r["not"] = "şişmiş satır çok az/çok"; return r

    def en_iyi(si, di):
        tpr = np.array([(di[si == 1] <= e).mean() for e in IZGARA]); fpr = np.array([(di[si == 0] <= e).mean() for e in IZGARA])
        j = int(np.argmax(tpr - fpr)); return IZGARA[j], tpr[j] - fpr[j]

    r["eps_yildiz"], r["youden"] = en_iyi(s, dd)
    r["AUC"] = float(roc_auc_score(s, -dd))
    rb = np.random.RandomState(9); b = []
    for _ in range(N_BOOT):
        i = rb.randint(0, len(s), len(s)); b.append(en_iyi(s[i], dd[i])[0])
    r["eps_alt"], r["eps_ust"] = float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))
    return r


def main(ad):
    yol = f"{M.OUT8}/checkpoint/faz8c_{ad}.pkl"
    SON = pickle.load(open(yol, "rb")) if os.path.exists(yol) else []
    tamam = {(r["bozma"], r["duzey"]) for r in SON}
    if all((t, d) in tamam for t, d in KOSUL):
        return
    t0 = time.time()
    X, y, v, _ = M.veri(ad)
    y = np.asarray(y).astype(str); v = np.asarray(v).astype(str)
    log(ad, "faz8c yüklendi", X.shape, f"{time.time() - t0:.0f}s")
    for tur, duzey in KOSUL:
        if (tur, duzey) in tamam:
            continue
        t1 = time.time()
        Xb, si, _ = boz(X, tur, duzey)
        r = kosul_olc(ad, Xb, y[si], v[si], tur, duzey)
        del Xb
        SON.append(r); pickle.dump(SON, open(yol + ".tmp", "wb")); os.replace(yol + ".tmp", yol)
        log(ad, "faz8c", {k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()}, f"{time.time() - t1:.0f}s")
    log(ad, "faz8c bitti", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        try:
            main(a)
        except Exception as e:
            import traceback
            log(a, "HATA", repr(e)); traceback.print_exc()
    tum = [r for f in sorted(os.listdir(f"{M.OUT8}/checkpoint")) if f.startswith("faz8c_")
           for r in pickle.load(open(f"{M.OUT8}/checkpoint/{f}", "rb"))]
    if tum:
        T = pd.DataFrame(tum)
        T.to_csv(f"{M.TAB8}/faz8c_bozma.csv", index=False)
        sys.path.insert(0, os.path.expanduser("~/work/pylibs"))
        c = [k for k in ("veri", "bozma", "duzey", "boyut", "sisme_payi", "tau0_benzeri", "tau_0.01", "tau_0.1", "d_medyan",
                         "eps_yildiz", "eps_alt", "eps_ust", "youden", "AUC", "not") if k in T]
        open(f"{M.OUT8}/FAZ8_BOZMA.md", "w", encoding="utf-8").write(
            "# Faz 8c — denetimli bozma: satır düzeyinde en iyi ε (otomatik)\n\n" + T[c].round(4).to_markdown(index=False) + "\n")
