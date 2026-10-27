"""Phase 7 H - reproduction and leakage audit of Arumugam & Ramalingam 2026 (Research Square, rs-11143924).

The authors' code (kaynak_kod/Code_NBaIoT, stage1-stage8) is followed exactly:
  stage1  6 random blocks x 500 rows from each file (default_rng(seed)), devices in sorted order; benign, mirai, gafgyt
  stage2  GroupShuffleSplit(test 0.2) by block id; correlation on the whole sample
  stage3  MI (40k subsample, k=3) + RF importance rank sum, |r|>=0.98 pruning, first 40 features
  stage4  HistGB(max_iter=300, max_depth=10, balanced), RobustScaler; LR(max_iter=500, balanced)
  stage8  RF probe (200 trees, depth 16, balanced): leave-one-device-out and leave-one-subtype-out
Our additions: fraction of test rows whose content appears in training; the same measures with the timestamp
columns removed and with leaking test rows dropped; signed-log transform for LR; multiple seeds.
Usage: python3 faz7_H_arumugam.py <tohum> [<tohum> ...]
"""
import sys
import time

from faz7_ortak import *  # noqa: F401,F403
from faz7_ortak import ck_oku, ck_yaz, icerik_hash, isaretli_log, log, satir_hash, yukle, zd_ozn
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler

N_JOBS = int(os.environ.get("FAZ7_NJOBS", "4"))


def orneklem(M, seed):
    """stage1_build_sample.py: file order and rng calls are identical to the authors' code."""
    rng = np.random.default_rng(seed)
    idx, blok = [], []
    for c in sorted(M.cihaz.unique()):
        dosyalar = [("benign", "benign")]
        for a in ("mirai", "gafgyt"):
            dosyalar += [(a, t) for t in sorted(M.tur[(M.cihaz == c) & (M.aile == a)].unique())]
        for a, t in dosyalar:
            satir = np.flatnonzero(((M.cihaz == c) & (M.aile == a) & (M.tur == t)).to_numpy())
            if len(satir) == 0:
                continue
            satir = satir[np.argsort(M.satir_no.to_numpy()[satir])]
            nb = len(satir) // 500
            sec = np.arange(nb) if nb <= 6 else np.sort(rng.choice(np.arange(nb), size=6, replace=False))
            for b in sec:
                idx.append(satir[b * 500:(b + 1) * 500])
                blok += [f"{c}|{a}|{t}|{b}"] * 500
    return np.concatenate(idx), np.array(blok)


def oznitelik_sec(Xtv, ytv, Xtum, ozn, seed):
    """stage3_feature_selection.py (the paper's K*=40 is used instead of the LightGBM knee-point scan)."""
    rng = np.random.default_rng(seed)
    alt = rng.choice(len(Xtv), size=min(40000, len(Xtv)), replace=False)
    mi = mutual_info_classif(Xtv[alt], ytv[alt], random_state=seed, n_neighbors=3)
    rf = RandomForestClassifier(n_estimators=150, max_depth=14, n_jobs=N_JOBS, random_state=seed,
                                class_weight="balanced").fit(Xtv[alt], ytv[alt])
    sira = pd.DataFrame({"f": range(len(ozn)), "mi": pd.Series(mi).rank(ascending=False),
                         "rf": pd.Series(rf.feature_importances_).rank(ascending=False)})
    sira = sira.assign(t=sira.mi + sira.rf).sort_values("t")
    with np.errstate(all="ignore"):
        kor = np.corrcoef(Xtum, rowvar=False)   # the authors compute the correlation on the whole sample, test included
    tut = []
    for f in sira.f:
        if not any(abs(kor[f, k]) >= 0.98 for k in tut):
            tut.append(int(f))
    return tut[:40]


def olc(y, p, siniflar, on):
    r = {f"{on}MCC": matthews_corrcoef(y, p), f"{on}F1": f1_score(y, p, average="macro")}
    for s in ("gafgyt:tcp", "gafgyt:udp", "gafgyt:scan", "mirai:scan"):
        m = y == s
        r[f"{on}geri_{s}"] = float((p[m] == s).mean()) if m.any() else np.nan
    return r


def main(seed):
    X, M, ozn = yukle()
    Hc = icerik_hash(X, ozn)                    # content (excluding timestamp)
    Hf = satir_hash(X, "float64")               # full row
    zi = set(zd_ozn(ozn))
    ad = f"H_{seed}"
    SON = ck_oku(ad)
    tamam = {r["kayit"] for r in SON}

    def ekle(r):
        SON.append(r); ck_yaz(ad, SON); tamam.add(r["kayit"]); log(seed, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    idx, blok = orneklem(M, seed)
    y = M.sinif.to_numpy()[idx]
    ybin = (y != "benign").astype(int)
    siniflar = sorted(set(y))
    Xs = np.asarray(X[idx])
    tv, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(Xs, ybin, groups=blok))
    assert not set(blok[tv]) & set(blok[te])

    # 1) leakage audit: does the test row appear verbatim in training (full row / content)
    if "sizinti" not in tamam:
        r = {"kayit": "sizinti", "seed": seed, "n": len(idx), "n_test": len(te),
             "tekrar_satir_ornek": int(pd.Series(Hf[idx]).duplicated().sum()),
             "tekrar_grup_ornek": int((pd.Series(Hf[idx]).value_counts() > 1).sum()),
             "test_egitimde_tam_%": 100 * np.isin(Hf[idx][te], Hf[idx][tv]).mean(),
             "test_egitimde_icerik_%": 100 * np.isin(Hc[idx][te], Hc[idx][tv]).mean()}
        for s in siniflar:
            m = y[te] == s
            r[f"icerik_sizinti_{s}_%"] = 100 * np.isin(Hc[idx][te][m], Hc[idx][tv]).mean()
        ekle(r)

    # 2) feature selection (the authors') and whether timestamp columns were selected
    if "secim" not in tamam:
        sec = oznitelik_sec(Xs[tv], y[tv], Xs, ozn, seed)
        ekle({"kayit": "secim", "seed": seed, "secilen": [ozn[i] for i in sec],
              "secilen_zd_sayisi": sum(i in zi for i in sec),
              "secilen_HHjit_sayisi": sum(ozn[i].startswith("HH_jit") for i in sec)})
    sec = [ozn.index(c) for c in next(r for r in SON if r["kayit"] == "secim")["secilen"]]
    setler = {"sec40": sec, "sec40_zdsiz": [i for i in sec if i not in zi],
              "hepsi_115": list(range(115)), "zd_yok_110": [i for i in range(115) if i not in zi]}
    temiz = ~np.isin(Hc[idx][te], Hc[idx][tv])   # test rows whose content is not present in training

    # 3) HistGB and LR, held-out test set: full and purged
    for oad, cols in setler.items():
        for mad in ("HistGB", "LR", "LR_log"):
            k = f"test|{mad}|{oad}"
            if k in tamam or (mad != "HistGB" and oad not in ("sec40", "sec40_zdsiz")):
                continue
            t0 = time.time()
            A, B = Xs[tv][:, cols], Xs[te][:, cols]
            if mad == "LR_log":
                A, B = isaretli_log(A), isaretli_log(B)
            if mad == "HistGB":
                mdl = HistGradientBoostingClassifier(max_iter=300, max_depth=10, random_state=seed, class_weight="balanced")
            else:
                mdl = LogisticRegression(max_iter=500, random_state=seed, class_weight="balanced")
            sc = RobustScaler().fit(A)
            mdl.fit(sc.transform(A), y[tv])
            p = mdl.predict(sc.transform(B))
            r = {"kayit": k, "seed": seed, "model": mad, "oznitelik": oad, "n_ozn": len(cols),
                 "test_n": len(te), "temiz_n": int(temiz.sum())}
            r.update(olc(y[te], p, siniflar, "tum_"))
            r.update(olc(y[te][temiz], p[temiz], siniflar, "temiz_"))
            r["sure_s"] = round(time.time() - t0, 1)
            ekle(r)

    # 4) stage8: leave-one-device-out (RF probe, whole sample) and leave-one-subtype-out
    def rf():
        return RandomForestClassifier(n_estimators=200, max_depth=16, n_jobs=N_JOBS, random_state=seed, class_weight="balanced")
    cihaz = M.cihaz.to_numpy()[idx]
    for oad in ("sec40", "sec40_zdsiz", "zd_yok_110"):
        cols = setler[oad]
        sc = RobustScaler().fit(Xs[tv][:, cols])
        Z = sc.transform(Xs[:, cols])
        for c in sorted(set(cihaz)):
            k = f"lodo|{oad}|{c}"
            if k in tamam:
                continue
            m = cihaz == c
            p = rf().fit(Z[~m], y[~m]).predict(Z[m])
            sz = np.isin(Hc[idx][m], Hc[idx][~m])
            r = {"kayit": k, "seed": seed, "oznitelik": oad, "cihaz": c, "test_n": int(m.sum()),
                 "icerik_sizinti_%": 100 * sz.mean(),
                 "MCC": matthews_corrcoef(y[m], p),
                 "MCC_temiz": matthews_corrcoef(y[m][~sz], p[~sz]) if (~sz).sum() > 10 else np.nan}
            ekle(r)
        for s in [s for s in siniflar if s != "benign"]:
            k = f"zeroday|{oad}|{s}"
            if k in tamam:
                continue
            m = y == s
            p = rf().fit(Z[~m], y[~m]).predict(Z[m])
            ekle({"kayit": k, "seed": seed, "oznitelik": oad, "alt_tur": s, "test_n": int(m.sum()),
                  "isaretlenen_%": 100 * float((p != "benign").mean()),
                  "en_cok_tahmin": pd.Series(p).value_counts().index[0]})
    log(f"H seed {seed} done")


if __name__ == "__main__":
    for s in sys.argv[1:]:
        main(int(s))
