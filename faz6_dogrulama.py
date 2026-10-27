"""Faz 6 — verification of claims and hypotheses (no TensorFlow).

Usage:  python3 faz6_dogrulama.py A|B|C|G   ·   python3 faz6_dogrulama.py D <seed>|ozet
  A  hash audit repeated with float64 (uniqueness, overlap across types/devices, S1-S5 leakage)
  B  timestamp artifact (which features, what each separates on its own)
  C  Faysal 2022 replication: Ecobee, random 75/25, RF and XGB, with/without HH_jit, 40-feature selection with RF, 5 seeds
  D  Faz 5 protocols (S1-S5) + leak-filtered S1f/S3f/S4f; feature set 115 / without HH_jit / HH_jit only
  G  CIC-IoT2023 replication and label-conflict audit (whether the finding is specific to N-BaIoT)

Output: faz6_ciktilar/tables/  ·  checkpoint: faz6_ciktilar/faz6_<part>_checkpoint.pkl
"""
import datetime as dt
import glob
import os
import pickle
import sys
import time
import warnings

_PYLIBS = os.path.expanduser("~/work/pylibs")
if os.path.isdir(_PYLIBS) and _PYLIBS not in sys.path:
    sys.path.insert(0, _PYLIBS)
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef

warnings.filterwarnings("ignore")

KOK = os.path.expanduser("~/work/IoT_Botnet/veri/nbaiot")
OUT = os.path.expanduser("~/work/IoT_Botnet/faz6_ciktilar")
TAB, CACHE = f"{OUT}/tables", f"{OUT}/cache"
for d in (TAB, CACHE):
    os.makedirs(d, exist_ok=True)

N_JOBS = int(os.environ.get("FAZ6_NJOBS", "8"))
UST = 150_000
MIRAI_KUME = {
    "Danmini_Doorbell": "K1", "Philips_B120N10_Baby_Monitor": "K1",
    "Ecobee_Thermostat": "K2",
    "SimpleHome_XCS7_1002_WHT_Security_Camera": "K2",
    "SimpleHome_XCS7_1003_WHT_Security_Camera": "K2",
    "Provision_PT_737E_Security_Camera": "K3",
    "Provision_PT_838_Security_Camera": "K3",
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ data
def yukle():
    """X (float64, mmap) and the meta table. On the first call it builds a cache from the CSVs."""
    xp, mp = f"{CACHE}/X64.npy", f"{CACHE}/meta.pkl"
    if not os.path.exists(xp):
        log("reading CSVs (float64)...")
        X, meta = [], []
        for f in sorted(glob.glob(f"{KOK}/**/*.csv", recursive=True)):
            if "demonstrate" in f:
                continue
            p = os.path.relpath(f, KOK).split(os.sep)
            c, a, t = (p[0], "benign", "benign") if len(p) == 2 else \
                      (p[0], p[1].replace("_attacks", ""), os.path.splitext(p[-1])[0])
            df = pd.read_csv(f)
            X.append(df.to_numpy(np.float64))
            meta.append(pd.DataFrame({"cihaz": c, "aile": a, "tur": t,
                                      "satir_no": np.arange(len(df), dtype=np.int32)}))
            ozn = list(df.columns)
        np.save(xp, np.concatenate(X))
        M = pd.concat(meta, ignore_index=True)
        M["sinif"] = np.where(M.aile == "benign", "benign", M.aile + ":" + M.tur)
        M["kotu"] = (M.aile != "benign").astype(np.int8)
        with open(mp, "wb") as fh:
            pickle.dump((M, ozn), fh)
    with open(mp, "rb") as fh:
        M, ozn = pickle.load(fh)
    return np.load(xp, mmap_mode="r"), M, ozn


def satir_hash(X, hassasiyet, haric=(), ad=None):
    """64-bit digest per row; column digests are combined with pandas hash_array.
    haric: indices of columns excluded from the digest (e.g. timestamp columns)."""
    yol = f"{CACHE}/hash_{ad or hassasiyet}.npy"
    if os.path.exists(yol):
        return np.load(yol)
    h = np.zeros(len(X), dtype=np.uint64)
    for j in (j for j in range(X.shape[1]) if j not in haric):
        s = np.ascontiguousarray(X[:, j]).astype(hassasiyet)
        h = (h * np.uint64(1099511628211)) ^ pd.util.hash_array(s)
    np.save(yol, h)
    return h


def zd_ozn(ozn):
    return [i for i, c in enumerate(ozn) if c.startswith("HH_jit_") and c.endswith("_mean")]


def utc(x):
    return dt.datetime.fromtimestamp(float(x), dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------------ splits (same definition as Faz 5)
def bolmeler(M, H, seed):
    rs = np.random.RandomState(seed)
    n = len(M)
    out = []
    p = rs.permutation(n); k = int(0.7 * n)
    out.append(("S1", "-", p[:k], p[k:]))
    tekil = pd.Series(H).drop_duplicates().index.to_numpy()
    p = rs.permutation(tekil); k = int(0.7 * len(p))
    out.append(("S2", "-", p[:k], p[k:]))
    tr, te = [], []
    for _, g in M.groupby(["cihaz", "sinif"], sort=False):
        g = g.sort_values("satir_no"); k = int(0.7 * len(g))
        tr.append(g.index[:k].to_numpy()); te.append(g.index[k:].to_numpy())
    out.append(("S3", "-", np.concatenate(tr), np.concatenate(te)))
    for c in sorted(M.cihaz.unique()):
        m = (M.cihaz == c).to_numpy()
        out.append(("S4", c, np.flatnonzero(~m), np.flatnonzero(m)))
    for kume in ("K1", "K2", "K3"):
        m = M.cihaz.isin([c for c, v in MIRAI_KUME.items() if v == kume]).to_numpy()
        out.append(("S5", kume, np.flatnonzero(~m), np.flatnonzero(m)))
    return out


def arindir(bl, H):
    """Removes from the test set the rows whose content hash is present in training (S1→S1f, S3→S3f, S4→S4f)."""
    out = []
    for b, k, tr, te in bl:
        if b in ("S1", "S3", "S4"):
            out.append((b + "f", k, tr, te[~np.isin(H[te], np.unique(H[tr]))]))
    return out


def sizinti(M, H, bl):
    aile = M.aile.to_numpy()
    rows = []
    for b, k, tr, te in bl:
        ici = np.isin(H[te], np.unique(H[tr]))
        r = {"bolme": b, "kat": k, "sizinti_tum_%": 100 * ici.mean()}
        for a in ("benign", "gafgyt", "mirai"):
            m = aile[te] == a
            r[f"sizinti_{a}_%"] = 100 * ici[m].mean() if m.any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ A
def parca_A():
    X, M, ozn = yukle()
    H = {p: satir_hash(X, p) for p in ("float64", "float32")}
    H["float64_zdsiz"] = satir_hash(X, "float64", haric=set(zd_ozn(ozn)), ad="float64_zdsiz")
    log("hash ready")
    anahtar = list(M.groupby(["cihaz", "aile", "tur"]).indices.items())

    t1 = []
    kume = {p: {} for p in H}
    for (c, a, t), idx in anahtar:
        r = {"cihaz": c, "aile": a, "tur": t, "satir": len(idx)}
        for p, h in H.items():
            u, say = np.unique(h[idx], return_counts=True)
            kume[p][(c, a, t)] = set(u.tolist())
            r[f"tekil_{p}"] = len(u)
            r[f"tekillik_{p}_%"] = 100 * len(u) / len(idx)
            r[f"baskin_pay_{p}_%"] = 100 * say.max() / len(idx)
        t1.append(r)
    t1 = pd.DataFrame(t1)
    t1.to_csv(f"{TAB}/A1_dosya_tekillik.csv", index=False)
    oz = t1[t1.aile != "benign"].groupby(["aile", "tur"])[["satir"] + [f"tekil_{p}" for p in H]].sum()
    for p in H: oz[f"tekillik_{p}_%"] = 100 * oz[f"tekil_{p}"] / oz.satir
    print(oz.round(3).to_string())

    # tcp/udp and benign-attack overlap
    t2, t4 = [], []
    for c in sorted(M.cihaz.unique()):
        for p in H:
            A, B = kume[p].get((c, "gafgyt", "tcp")), kume[p].get((c, "gafgyt", "udp"))
            if A is not None:
                t2.append({"cihaz": c, "hassasiyet": p, "ortak_tekil": len(A & B)})
            ben = kume[p][(c, "benign", "benign")]
            ortak = sum(len(ben & s) for (cc, a, _), s in kume[p].items() if cc == c and a != "benign")
            t4.append({"cihaz": c, "hassasiyet": p, "iyi_saldiri_ortak_tekil": ortak})
    pd.DataFrame(t2).to_csv(f"{TAB}/A2_tcp_udp_ortak.csv", index=False)
    pd.DataFrame(t4).to_csv(f"{TAB}/A4_iyi_saldiri_ortak.csv", index=False)
    print(pd.DataFrame(t2).pivot(index="cihaz", columns="hassasiyet", values="ortak_tekil").to_string())
    print(pd.DataFrame(t4).groupby("hassasiyet").iyi_saldiri_ortak_tekil.sum().to_string())

    # cross-device overlap
    t3 = []
    for a, t in sorted({(a, t) for (_, a, t) in kume["float64"] if a != "benign"}):
        cs = sorted(c for (c, aa, tt) in kume["float64"] if (aa, tt) == (a, t))
        for i, ca in enumerate(cs):
            for cb in cs[i + 1:]:
                r = {"sinif": f"{a}:{t}", "cihaz_a": ca, "cihaz_b": cb}
                for p in H:
                    A, B = kume[p][(ca, a, t)], kume[p][(cb, a, t)]
                    r[f"pay_{p}_%"] = 100 * len(A & B) / min(len(A), len(B))
                t3.append(r)
    t3 = pd.DataFrame(t3)
    t3.to_csv(f"{TAB}/A3_cihazlar_arasi.csv", index=False)
    print(t3.groupby("sinif")[[f"pay_{p}_%" for p in H]].agg(["min", "mean"]).round(1).to_string())

    # leakage
    rows = []
    for p, h in H.items():
        s = sizinti(M, h, bolmeler(M, h, 42)); s.insert(0, "hassasiyet", p); rows.append(s)
    s = pd.concat(rows)
    s.to_csv(f"{TAB}/A5_sizinti.csv", index=False)
    print(s.groupby(["hassasiyet", "bolme"])[[c for c in s.columns if c.startswith("sizinti")]].mean().round(2).to_string())


# ------------------------------------------------------------------ B
def parca_B():
    X, M, ozn = yukle()
    aile, sinif = M.aile.to_numpy(), M.sinif.to_numpy()

    # B1: which columns carry timestamp values
    rows = []
    for j, c in enumerate(ozn):
        buyuk = np.asarray(X[:, j]) > 1e9
        if buyuk.any():
            r = {"oznitelik": c}
            for s in sorted(set(sinif)):
                r[s] = 100 * buyuk[sinif == s].mean()
            rows.append(r)
    b1 = pd.DataFrame(rows)
    b1.to_csv(f"{TAB}/B1_zd_sutunlar.csv", index=False)
    print(b1.set_index("oznitelik").round(1).T.to_string())

    # B2: the "HH_jit_*_mean > 1e9" rule as an attack detector on its own
    zi = zd_ozn(ozn)
    kural = (np.asarray(X[:, zi]) > 1e9).any(axis=1)
    b2 = pd.DataFrame({"sinif": sinif, "isaret": kural}).groupby("sinif").isaret.mean().mul(100)
    b2.to_csv(f"{TAB}/B2_kural_dedektor.csv")
    print(b2.round(2).to_string())
    y = M.kotu.to_numpy()
    print(f"rule: TPR={kural[y == 1].mean():.4f}  FPR={kural[y == 0].mean():.5f}  MCC={matthews_corrcoef(y, kural):.4f}")

    # B3: date range of the timestamps
    ts = np.asarray(X[:, zi[0]])
    rows = []
    for (c, s), idx in M.groupby(["cihaz", "sinif"]).indices.items():
        v = ts[idx][ts[idx] > 1e9]
        if len(v):
            rows.append({"cihaz": c, "sinif": s, "zd_satir": len(v),
                         "ilk": utc(v.min()), "son": utc(v.max()), "tekil_zd": len(np.unique(v))})
    b3 = pd.DataFrame(rows)
    b3.to_csv(f"{TAB}/B3_zd_tarihleri.csv", index=False)
    print(b3[b3.sinif == "benign"].to_string(index=False))
    print(b3.groupby("sinif")[["ilk", "son"]].agg({"ilk": "min", "son": "max"}).to_string())

    # B4: what can be learned with only the 5 HH_jit_*_mean features (S1 random, 3 seeds)
    rows = []
    for seed in (42, 7, 123):
        rs = np.random.RandomState(seed)
        for hedef, maske in (("sinif", np.ones(len(M), bool)),
                             ("mirai_kume", (aile == "mirai")),
                             ("cihaz_benign", (aile == "benign"))):
            idx = np.flatnonzero(maske)
            yv = (M.cihaz.map(MIRAI_KUME).to_numpy() if hedef == "mirai_kume"
                  else M.cihaz.to_numpy() if hedef == "cihaz_benign" else sinif)
            p = rs.permutation(idx); k = int(0.7 * len(p))
            tr, te = p[:k][:UST], p[k:][:UST]
            Xz = np.asarray(X[:, zi])
            m = RandomForestClassifier(n_estimators=100, n_jobs=N_JOBS, random_state=seed)
            m.fit(Xz[tr], yv[tr]); pr = m.predict(Xz[te])
            r = {"seed": seed, "hedef": hedef, "dogruluk": float((pr == yv[te]).mean()),
                 "F1_makro": f1_score(yv[te], pr, average="macro")}
            if hedef == "sinif":
                for s in ("gafgyt:tcp", "gafgyt:udp", "gafgyt:scan", "mirai:scan", "benign"):
                    mm = yv[te] == s
                    r[f"geri_cagirma_{s}"] = float((pr[mm] == s).mean())
            rows.append(r); log(r)
    b4 = pd.DataFrame(rows)
    b4.to_csv(f"{TAB}/B4_yalniz_zd_ozn.csv", index=False)
    print(b4.groupby("hedef").mean(numeric_only=True).round(4).T.to_string())


# ------------------------------------------------------------------ C
def xgb(seed):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.3, tree_method="hist",
                         n_jobs=N_JOBS, random_state=seed)


def parca_C():
    X, M, ozn = yukle()
    from sklearn.preprocessing import LabelEncoder
    eco = np.flatnonzero((M.cihaz == "Ecobee_Thermostat").to_numpy())
    siniflar = M.sinif.to_numpy()[eco]
    le = LabelEncoder().fit(siniflar)
    y_all = le.transform(siniflar)
    zi = zd_ozn(ozn)
    jit_hepsi = [i for i, c in enumerate(ozn) if c.startswith("HH_jit_")]
    setler = {"hepsi_115": list(range(len(ozn))),
              "HHjit_mean_yok_110": [i for i in range(len(ozn)) if i not in zi],
              "HHjit_hepsi_yok_100": [i for i in range(len(ozn)) if i not in jit_hepsi]}
    # Subsample matching Faysal's class counts (Kaggle copy) and the full Ecobee
    faysal = {"benign": 13113, "mirai:ack": 27188, "mirai:scan": 9502, "mirai:syn": 23361,
              "mirai:udp": 15148, "mirai:udpplain": 26210, "gafgyt:combo": 21205,
              "gafgyt:junk": 24250, "gafgyt:scan": 21995, "gafgyt:tcp": 23755, "gafgyt:udp": 24102}
    ck = f"{OUT}/faz6_C_checkpoint.pkl"
    SON = pickle.load(open(ck, "rb")) if os.path.exists(ck) else []
    tamam = {(r["veri"], r["oznitelik"], r["model"], r["seed"]) for r in SON}
    tcp, udp = le.transform(["gafgyt:tcp"])[0], le.transform(["gafgyt:udp"])[0]
    Xe = np.asarray(X[eco])
    for seed in range(5):
        rs = np.random.RandomState(seed)
        alt = np.concatenate([rs.choice(np.flatnonzero(siniflar == s), n, replace=False)
                              for s, n in faysal.items()])
        for veri, idx in (("faysal_alt", alt), ("ecobee_tam", np.arange(len(eco)))):
            p = rs.permutation(idx); k = int(0.75 * len(p))
            tr, te = p[:k], p[k:]
            sec40 = None
            for oad, cols in setler.items():
                for mad in ("RF", "XGB", "RF_sec40", "XGB_sec40"):
                    if mad.endswith("sec40") and oad != "hepsi_115":
                        continue
                    if (veri, oad, mad, seed) in tamam:
                        continue
                    t0 = time.time()
                    if mad.endswith("sec40"):
                        # Faysal: top 40 features by RF feature importance, then the classifier
                        if sec40 is None:
                            imp = RandomForestClassifier(n_estimators=100, n_jobs=N_JOBS, random_state=seed)
                            imp.fit(Xe[tr], y_all[tr])
                            sec40 = np.argsort(imp.feature_importances_)[::-1][:40]
                        cols = sec40
                    m = (RandomForestClassifier(n_estimators=100, n_jobs=N_JOBS, random_state=seed)
                         if mad.startswith("RF") else xgb(seed))
                    m.fit(Xe[tr][:, cols], y_all[tr]); pr = m.predict(Xe[te][:, cols])
                    yt = y_all[te]
                    r = {"veri": veri, "oznitelik": oad, "model": mad, "seed": seed,
                         "dogruluk": float((pr == yt).mean()),
                         "tcp_geri": float((pr[yt == tcp] == tcp).mean()),
                         "udp_geri": float((pr[yt == udp] == udp).mean()),
                         "tcp_to_udp": float((pr[yt == tcp] == udp).mean()),
                         "diger_9_min_geri": min(float((pr[yt == c] == c).mean())
                                                 for c in range(len(le.classes_)) if c not in (tcp, udp)),
                         "secilen_HHjit_sayisi": int(sum(ozn[i].startswith("HH_jit_") for i in cols)),
                         "sure_s": round(time.time() - t0, 1)}
                    SON.append(r); log(r)
                    pickle.dump(SON, open(ck, "wb"))
    R = pd.DataFrame(SON)
    R.to_csv(f"{TAB}/C_faysal_tekrar.csv", index=False)
    print(R.groupby(["veri", "oznitelik", "model"])[["dogruluk", "tcp_geri", "udp_geri", "tcp_to_udp", "diger_9_min_geri", "secilen_HHjit_sayisi"]]
          .agg(["mean", "std"]).round(4).to_string())


# ------------------------------------------------------------------ D
def parca_D(arg):
    if arg == "ozet":
        R = pd.concat([pd.DataFrame(pickle.load(open(f, "rb")))
                       for f in sorted(glob.glob(f"{OUT}/faz6_D_*_checkpoint.pkl"))], ignore_index=True)
        R.to_csv(f"{TAB}/D_protokoller.csv", index=False)
        met = [c for c in R.columns if c not in ("bolme", "kat", "model", "oznitelik", "seed", "sure_s", "test_n")]
        print(R.groupby(["bolme", "model", "oznitelik"])[met].mean().round(4).to_string())
        return
    seed = int(arg)
    X, M, ozn = yukle()
    # S2 deduplication by content hash (excluding timestamp columns; equivalent to Faz 5's float32 hash)
    H = satir_hash(X, "float64", haric=set(zd_ozn(ozn)), ad="float64_zdsiz")
    jit = [i for i, c in enumerate(ozn) if c.startswith("HH_jit_")]
    setler = {"hepsi_115": np.arange(len(ozn)),
              "HHjit_hepsi_yok_100": np.array([i for i in range(len(ozn)) if i not in jit]),
              "yalniz_HHjit_15": np.array(jit)}
    y_bin, y_cok, aile = M.kotu.to_numpy(), M.sinif.to_numpy(), M.aile.to_numpy()
    ZOR = ("gafgyt:tcp", "gafgyt:udp", "gafgyt:junk")
    ck = f"{OUT}/faz6_D_{seed}_checkpoint.pkl"
    eski = f"{OUT}/faz6_D_checkpoint.pkl"   # carry-over from the single-file first version
    SON = pickle.load(open(ck, "rb")) if os.path.exists(ck) else \
        [r for r in (pickle.load(open(eski, "rb")) if os.path.exists(eski) else []) if r["seed"] == seed]
    tamam = {(r["bolme"], r["kat"], r["model"], r["oznitelik"], r["seed"]) for r in SON}
    modeller = {
        "RF": lambda s: RandomForestClassifier(n_estimators=150, n_jobs=N_JOBS, random_state=s),
        "HistGB": lambda s: HistGradientBoostingClassifier(max_iter=200, random_state=s),
    }
    bl = bolmeler(M, H, seed)
    for b, k, tr, te in bl + arindir(bl, H):
            rs = np.random.RandomState(seed)
            tr_s = tr if len(tr) <= UST else rs.choice(tr, UST, replace=False)
            te_s = te if len(te) <= UST else rs.choice(te, UST, replace=False)
            for oad, cols in setler.items():
                Xtr, Xte = np.asarray(X[tr_s])[:, cols], np.asarray(X[te_s])[:, cols]
                for mad, fab in modeller.items():
                    if oad == "yalniz_HHjit_15" and mad != "RF":
                        continue
                    if (b, k, mad, oad, seed) in tamam:
                        continue
                    t0 = time.time()
                    mb = fab(seed).fit(Xtr, y_bin[tr_s]); pb = mb.predict(Xte)
                    mc = fab(seed).fit(Xtr, y_cok[tr_s]); pc = mc.predict(Xte)
                    yb, yc, at = y_bin[te_s], y_cok[te_s], aile[te_s]
                    r = {"bolme": b, "kat": k, "model": mad, "oznitelik": oad, "seed": seed, "test_n": len(te),
                         "MCC_ikili": matthews_corrcoef(yb, pb),
                         "F1_makro_cok": f1_score(yc, pc, average="macro"),
                         "yanlis_alarm": float((pb[at == "benign"] == 1).mean())}
                    for a in ("gafgyt", "mirai"):
                        m = at == a
                        r[f"yakalama_{a}"] = float((pb[m] == 1).mean()) if m.any() else np.nan
                        r[f"tur_dogrulugu_{a}"] = float((pc[m] == yc[m]).mean()) if m.any() else np.nan
                    for s in ZOR:
                        m = yc == s
                        r[f"geri_{s}"] = float((pc[m] == s).mean()) if m.any() else np.nan
                        r[f"ikili_yakalama_{s}"] = float((pb[m] == 1).mean()) if m.any() else np.nan
                    r["sure_s"] = round(time.time() - t0, 1)
                    SON.append(r); log(b, k, mad, oad, seed, f"MCC={r['MCC_ikili']:.4f} F1={r['F1_makro_cok']:.4f} ({r['sure_s']}s)")
                    pickle.dump(SON, open(ck, "wb"))
    log(f"D seed {seed} bitti: {len(SON)} rows")


# ------------------------------------------------------------------ G
GBOZUK = set()   # files to read with the python engine if needed


def parca_G():
    """CIC-IoT2023: repetition within a label, conflict across labels (same vector, different label), random-split leakage."""
    kok = os.path.expanduser("~/work/IoT_Botnet/veri/cic2023/merged/MERGED_CSV")
    hs, ls, bozuk = [], [], {}
    for f in sorted(glob.glob(f"{kok}/*.csv")):
        atla = []
        df = pd.read_csv(f, engine="python", on_bad_lines=lambda r: atla.append(r) and None) \
            if f in GBOZUK else pd.read_csv(f, engine="pyarrow", on_bad_lines="skip")
        with open(f, "rb") as fh:
            satir = sum(1 for _ in fh) - 1
        if satir != len(df):
            bozuk[os.path.basename(f)] = satir - len(df)
        y = df.pop("Label").astype(str).to_numpy()
        h = np.zeros(len(df), dtype=np.uint64)
        for c in df.columns:
            h = (h * np.uint64(1099511628211)) ^ pd.util.hash_array(df[c].to_numpy(np.float64))
        hs.append(h); ls.append(y)
        log(os.path.basename(f), len(df))
    H, Y = np.concatenate(hs), np.concatenate(ls)
    log("corrupt/skipped rows:", bozuk or "none")
    pd.Series(bozuk, dtype=int).to_csv(f"{TAB}/G0_cic2023_bozuk_satir.csv")
    n = len(H)
    rows = []
    for lab in np.unique(Y):
        m = Y == lab
        u, say = np.unique(H[m], return_counts=True)
        rows.append({"etiket": lab, "satir": int(m.sum()), "tekil": len(u),
                     "tekillik_%": 100 * len(u) / m.sum(), "baskin_pay_%": 100 * say.max() / m.sum()})
    g1 = pd.DataFrame(rows).sort_values("tekillik_%")
    g1.to_csv(f"{TAB}/G1_cic2023_etiket_tekillik.csv", index=False)
    print(g1.to_string(index=False))
    # same vector under more than one label
    d = pd.DataFrame({"h": H, "y": Y}).drop_duplicates()
    cok = d.groupby("h").y.nunique()
    celiski = cok[cok > 1].index
    print(f"unique vectors: {len(cok):,}  appearing under more than one label: {len(celiski):,}  "
          f"share of rows carrying these vectors: %{100 * np.isin(H, celiski.to_numpy()).mean():.3f}")
    # random 70/30 split leakage (3 seeds)
    s = []
    for seed in (42, 7, 123):
        p = np.random.RandomState(seed).permutation(n); k = int(0.7 * n)
        ici = np.isin(H[p[k:]], np.unique(H[p[:k]]))
        s.append({"seed": seed, "sizinti_tum_%": 100 * ici.mean(),
                  "sizinti_benign_%": 100 * ici[Y[p[k:]] == "BENIGN"].mean()})
    g2 = pd.DataFrame(s)
    g2.to_csv(f"{TAB}/G2_cic2023_sizinti.csv", index=False)
    print(g2.round(3).to_string(index=False))
    pd.Series({"satir": n, "tekil": int(len(cok)), "celiskili_vektor": int(len(celiski)),
               "celiskili_satir_%": float(100 * np.isin(H, celiski.to_numpy()).mean())}).to_csv(f"{TAB}/G3_cic2023_ozet.csv")


if __name__ == "__main__":
    if sys.argv[1] == "D":
        parca_D(sys.argv[2])
    else:
        {"A": parca_A, "B": parca_B, "C": parca_C, "G": parca_G}[sys.argv[1]]()
