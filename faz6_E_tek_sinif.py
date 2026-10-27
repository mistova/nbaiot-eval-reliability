"""Faz 6 / E — testing claims 8 and 9 in the one-class regime (TensorFlow; no parallel sklearn routine).

Same design as Faz 4b: float32 reading, benign time order 60%/10%/30%, mu/sd from the
calibration slice, clipping at +-20, q99 threshold, first 30,000 rows per attack type.
Variables: configuration (D-D per-device model / G-G global model) x feature set
(115 / 100 without HH_jit) x 5 seeds.

Extra measurement: error and position relative to the threshold of the benign "first packet" rows
that match the dominant content vector of tcp/udp (H: the tcp TPR is determined by whether this single point crosses the threshold).

Output: faz6_ciktilar/tables/E_tek_sinif.csv · checkpoint: faz6_ciktilar/faz6_E_checkpoint.pkl
"""
import glob
import os
import pickle
import random
import sys
import time
import warnings
from collections import defaultdict

_PYLIBS = os.path.expanduser("~/work/pylibs")
if os.path.isdir(_PYLIBS) and _PYLIBS not in sys.path:
    sys.path.insert(0, _PYLIBS)
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "8"
os.environ["CUDA_CACHE_PATH"] = os.path.expanduser("~/work/.cuda_cache")

import ctypes
try:
    import nvidia
    libdirs = sorted(glob.glob(os.path.join(os.path.dirname(nvidia.__file__), "*", "lib")))
    os.environ["LD_LIBRARY_PATH"] = ":".join(libdirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    for d in libdirs:
        for so in sorted(glob.glob(os.path.join(d, "*.so*"))):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
except ImportError:
    pass

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
KOK = os.path.expanduser("~/work/IoT_Botnet/veri/nbaiot")
OUT = os.path.expanduser("~/work/IoT_Botnet/faz6_ciktilar")
CKPT = f"{OUT}/faz6_E_checkpoint.pkl"
TOHUMLAR = [42, 7, 2024, 13, 101]
EPOCHS, BATCH, PATIENCE, LR, KOD = 60, 256, 8, 1e-3, 16
KALIBRASYON, ESIK_DILIM, SALDIRI_TEST = 0.60, 0.10, 30_000
ZOR = ["gafgyt:tcp", "gafgyt:udp"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)


def ae(boyut, tohum):
    set_seed(tohum)
    i = layers.Input(shape=(boyut,))
    x = layers.Dense(64, activation="relu")(i)
    x = layers.Dense(32, activation="relu")(x)
    k = layers.Dense(KOD, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(k)
    x = layers.Dense(64, activation="relu")(x)
    m = models.Model(i, layers.Dense(boyut, activation="linear")(x))
    m.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="mse")
    return m


def egit(kal, tohum):
    mu, sd = kal.mean(0), kal.std(0) + 1e-8
    X = np.clip((kal - mu) / sd, -20, 20).astype("float32")
    nv = max(1, int(0.1 * len(X)))
    m = ae(X.shape[1], tohum)
    m.fit(X[:-nv], X[:-nv], validation_data=(X[-nv:], X[-nv:]), epochs=EPOCHS, batch_size=BATCH,
          verbose=0, callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                                        restore_best_weights=True)])
    return m, mu, sd


def hata(m, mu, sd, A):
    X = np.clip((A - mu) / sd, -20, 20).astype("float32")
    R = m.predict(X, batch_size=4096, verbose=0)
    return np.mean((X - R) ** 2, axis=1)


def main():
    dosyalar = [f for f in sorted(glob.glob(f"{KOK}/**/*.csv", recursive=True)) if "demonstrate" not in f]
    IYI, SAL = {}, defaultdict(dict)
    for f in dosyalar:
        p = os.path.relpath(f, KOK).split(os.sep)
        df = pd.read_csv(f, dtype="float32")
        if len(p) == 2:
            IYI[p[0]] = df.to_numpy()
        else:
            SAL[p[0]][f"{p[1].replace('_attacks', '')}:{os.path.splitext(p[-1])[0]}"] = df.to_numpy()[:SALDIRI_TEST]
    ozn = list(pd.read_csv(dosyalar[0], nrows=1).columns)
    jit = [i for i, c in enumerate(ozn) if c.startswith("HH_jit_")]
    zd = [i for i, c in enumerate(ozn) if c.startswith("HH_jit_") and c.endswith("_mean")]
    setler = {"hepsi_115": np.arange(len(ozn)),
              "HHjit_hepsi_yok_100": np.array([i for i in range(len(ozn)) if i not in jit])}
    CIHAZ = sorted(IYI)

    def bol(A):
        n = len(A); i1 = int(KALIBRASYON * n); i2 = int((KALIBRASYON + ESIK_DILIM) * n)
        return A[:i1], A[i1:i2], A[i2:]
    BOL = {c: dict(zip(("kal", "esik", "test"), bol(IYI[c]))) for c in CIHAZ}

    # benign rows identical to the tcp dominant content vector (excluding the timestamp)
    def icerik(A):
        return pd.util.hash_pandas_object(pd.DataFrame(np.delete(A, zd, axis=1)), index=False).to_numpy()
    ILK = {}
    for c in CIHAZ:
        ht = icerik(SAL[c]["gafgyt:tcp"]); u, n = np.unique(ht, return_counts=True)
        ILK[c] = IYI[c][icerik(IYI[c]) == u[n.argmax()]]
    log("loaded;", {c[:10]: len(v) for c, v in ILK.items()}, "first-packet rows")

    D = pickle.load(open(CKPT, "rb")) if os.path.exists(CKPT) else []
    tamam = {(r["yapilandirma"], r["cihaz"], r["oznitelik"], r["tohum"]) for r in D}

    def esikler(e):
        kes = np.quantile(e, 0.99); le = np.log10(e + 1e-12)
        return {"q99_esik": float(kes), "q99_budanmis": float(np.quantile(e[e <= kes], 0.99)),
                "log_q99": float(10 ** (le.mean() + 2.33 * le.std()))}

    def kaydet(yap, c, oad, tohum, h_esik, h_test, h_sal, h_ilk):
        auc = {t: float(roc_auc_score(np.r_[np.zeros(len(h_test)), np.ones(len(h_sal[t]))],
                                      np.r_[h_test, h_sal[t]])) for t in ZOR if t in h_sal}
        for sad, e in esikler(h_esik).items():
            r = {"yapilandirma": yap, "cihaz": c, "oznitelik": oad, "tohum": tohum, "strateji": sad, "esik": e,
                 "FPR": float((h_test > e).mean()),
                 "TPR_kolay": float(np.mean(np.concatenate([v > e for t, v in h_sal.items() if t not in ZOR]))),
                 "ilk_paket_hata_med": float(np.median(h_ilk)) if len(h_ilk) else np.nan,
                 "ilk_paket_esigi_gecer": float((h_ilk > e).mean()) if len(h_ilk) else np.nan}
            for t in ZOR:
                if t in h_sal:
                    r[f"TPR_{t}"] = float((h_sal[t] > e).mean())
                    r[f"AUC_{t}"] = auc[t]
                    r[f"tekil_hata_{t}"] = int(len(np.unique(np.round(h_sal[t], 6))))
            D.append(r)
        pickle.dump(D, open(CKPT, "wb"))
        log(yap, c[:24], oad, tohum, f"tcp={r.get('TPR_gafgyt:tcp', np.nan):.3f} AUC_tcp={auc.get('gafgyt:tcp', np.nan):.3f} "
            f"ilk_paket_gecer={r['ilk_paket_esigi_gecer']:.2f} FPR={r['FPR']:.4f} (son strateji: {sad})")

    for tohum in TOHUMLAR:
        for oad, cols in setler.items():
            for c in CIHAZ:
                if ("D-D", c, oad, tohum) in tamam:
                    continue
                m, mu, sd = egit(BOL[c]["kal"][:, cols], tohum)
                kaydet("D-D", c, oad, tohum, hata(m, mu, sd, BOL[c]["esik"][:, cols]),
                       hata(m, mu, sd, BOL[c]["test"][:, cols]),
                       {t: hata(m, mu, sd, A[:, cols]) for t, A in SAL[c].items()},
                       hata(m, mu, sd, ILK[c][:, cols]))
                del m; tf.keras.backend.clear_session()
            if all(("G-G", c, oad, tohum) in tamam for c in CIHAZ):
                continue
            m, mu, sd = egit(np.concatenate([BOL[c]["kal"][:, cols] for c in CIHAZ]), tohum)
            h_esik_g = np.concatenate([hata(m, mu, sd, BOL[c]["esik"][:, cols]) for c in CIHAZ])
            for c in CIHAZ:
                kaydet("G-G", c, oad, tohum, h_esik_g, hata(m, mu, sd, BOL[c]["test"][:, cols]),
                       {t: hata(m, mu, sd, A[:, cols]) for t, A in SAL[c].items()},
                       hata(m, mu, sd, ILK[c][:, cols]))
            del m; tf.keras.backend.clear_session()

    R = pd.DataFrame(D)
    R.to_csv(f"{OUT}/tables/E_tek_sinif.csv", index=False)
    print(R.groupby(["yapilandirma", "oznitelik", "strateji"])[["TPR_gafgyt:tcp", "TPR_gafgyt:udp", "AUC_gafgyt:tcp",
                                                               "TPR_kolay", "FPR", "ilk_paket_esigi_gecer"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
