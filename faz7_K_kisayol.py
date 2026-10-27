"""Phase 7 K - shortcut scan (GPU XGBoost).

K1  11 classes from a single feature (D'hooge 2022 single-feature test; Flood 2024 HDF_C heuristic): S1 random and S6f
    (block-disjoint, leakage-purged). Macro F1, MCC, tcp/udp recall for each feature.
K2  Feature groups (MI_dir, H, HH, HH_jit, HpHp) and lambda windows; group alone / group excluded.
K3  Device identifiability: predict the device from rows of the same attack type. If the capture is shared across
    devices, accuracy falls to chance level; it should stay high for benign. (with/without timestamp)
Usage: python3 faz7_K_kisayol.py
"""
import re

from faz7_ortak import *  # noqa: F401,F403
from faz7_ortak import arindir7, bolmeler7, ck_oku, ck_yaz, icerik_hash, log, yukle, zd_ozn
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
from xgboost import XGBClassifier

N = 150_000


def xgb(seed, n_est=100, depth=6):
    return XGBClassifier(n_estimators=n_est, max_depth=depth, learning_rate=0.2, tree_method="hist",
                         device="cuda", random_state=seed)


def main():
    X, M, ozn = yukle()
    H = icerik_hash(X, ozn)
    zi = set(zd_ozn(ozn))
    y = M.sinif.to_numpy()
    siniflar = np.array(sorted(set(y)))
    yk = np.searchsorted(siniflar, y)
    SON = ck_oku("K")
    tamam = {r["kayit"] for r in SON}

    def ekle(r):
        SON.append(r); ck_yaz("K", SON); tamam.add(r["kayit"]); log(r["kayit"], {k: round(v, 4) for k, v in r.items() if isinstance(v, float)})

    def degerlendir(kayit, cols, tr, te, seed=42, **ek):
        if kayit in tamam:
            return
        m = xgb(seed).fit(np.asarray(X[tr])[:, cols].astype(np.float32), yk[tr])
        p = m.predict(np.asarray(X[te])[:, cols].astype(np.float32))
        yt = yk[te]
        r = {"kayit": kayit, **ek, "F1_makro": f1_score(yt, p, average="macro"), "MCC": matthews_corrcoef(yt, p),
             "dogruluk": float((p == yt).mean())}
        for s in ("gafgyt:tcp", "gafgyt:udp", "gafgyt:scan", "mirai:scan", "benign"):
            i = int(np.searchsorted(siniflar, s)); mm = yt == i
            r[f"geri_{s}"] = float((p[mm] == i).mean()) if mm.any() else np.nan
        ekle(r)

    rs = np.random.RandomState(42)
    bl = bolmeler7(M, H, 42)
    S1 = next(x for x in bl if x[0] == "S1")
    S6f = next(x for x in arindir7(bl, H) if x[0] == "S6f")
    bolmeler_ = {"S1": (rs.choice(S1[2], N, replace=False), rs.choice(S1[3], N, replace=False)),
                 "S6f": (rs.choice(S6f[2], N, replace=False), rs.choice(S6f[3], min(N, len(S6f[3])), replace=False))}

    # K1 single feature
    for bad, (tr, te) in bolmeler_.items():
        for j, c in enumerate(ozn):
            degerlendir(f"K1|{bad}|{c}", [j], tr, te, bolme=bad, oznitelik=c, zd=j in zi)

    # K2 groups
    grup = {}
    for j, c in enumerate(ozn):
        g = re.match(r"(MI_dir|HH_jit|HpHp|HH|H)_L([\d.]+)_", c)
        grup.setdefault(("grup", g.group(1)), []).append(j)
        grup.setdefault(("lambda", g.group(2)), []).append(j)
    for bad, (tr, te) in bolmeler_.items():
        for (tip, g), cols in grup.items():
            degerlendir(f"K2|{bad}|{tip}|{g}|yalniz", cols, tr, te, bolme=bad, tip=tip, grup=g, kip="yalniz")
            degerlendir(f"K2|{bad}|{tip}|{g}|haric", [j for j in range(len(ozn)) if j not in cols], tr, te,
                        bolme=bad, tip=tip, grup=g, kip="haric")
            degerlendir(f"K2|{bad}|{tip}|{g}|yalniz_zdsiz", [j for j in cols if j not in zi], tr, te,
                        bolme=bad, tip=tip, grup=g, kip="yalniz_zdsiz") if any(j in zi for j in cols) else None

    # K3 device identifiability (random 70/30, per class)
    cihaz = M.cihaz.to_numpy()
    cs = np.array(sorted(set(cihaz)))
    ck = np.searchsorted(cs, cihaz)
    setler = {"hepsi_115": list(range(len(ozn))), "zd_yok_110": [j for j in range(len(ozn)) if j not in zi]}
    for s in siniflar:
        idx = np.flatnonzero(y == s)
        for oad, cols in setler.items():
            for seed in (42, 7, 123):
                kayit = f"K3|{s}|{oad}|{seed}"
                if kayit in tamam:
                    continue
                r2 = np.random.RandomState(seed)
                p = r2.permutation(idx)[:2 * N]; k = int(0.7 * len(p))
                tr, te = p[:k], p[k:]
                var = np.unique(ck[tr]); har = np.full(len(cs), -1); har[var] = np.arange(len(var))
                m = xgb(seed).fit(np.asarray(X[tr])[:, cols].astype(np.float32), har[ck[tr]])
                pr = var[m.predict(np.asarray(X[te])[:, cols].astype(np.float32)).astype(int)]
                say = np.bincount(ck[te], minlength=len(cs))
                ekle({"kayit": kayit, "sinif": s, "oznitelik": oad, "seed": seed, "cihaz_sayisi": int(len(var)),
                      "dogruluk": float((pr == ck[te]).mean()), "sans_cogunluk": float(say.max() / say.sum()),
                      "F1_makro": f1_score(ck[te], pr, average="macro")})
    pd.DataFrame(SON).to_csv(f"{TAB}/K_kisayol.csv", index=False)
    log("K done")


if __name__ == "__main__":
    main()
