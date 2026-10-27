"""Faz 7 Q — Extension of Faz 6 D (CPU: RF, HistGB).

New: S6 (disjoint 500-row block split, Arumugam 2026) and S6f; the 'zd_yok_110' set with only the 5 timestamp
columns removed; 5 seeds; recall for all classes; share of test rows found in training as content.
Usage: python3 faz7_Q_protokoller.py <seed>
"""
import sys
import time

from faz7_ortak import *  # noqa: F401,F403
from faz7_ortak import UST, arindir7, bolmeler7, ck_oku, ck_yaz, icerik_hash, log, oznitelik_setleri, yukle
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef

N_JOBS = int(os.environ.get("FAZ7_NJOBS", "4"))


def main(seed):
    X, M, ozn = yukle()
    H = icerik_hash(X, ozn)
    setler = oznitelik_setleri(ozn)
    y_bin, y_cok, aile = M.kotu.to_numpy(), M.sinif.to_numpy(), M.aile.to_numpy()
    siniflar = sorted(set(y_cok))
    ad = f"Q_{seed}"
    SON = ck_oku(ad)
    tamam = {(r["bolme"], r["kat"], r["model"], r["oznitelik"]) for r in SON}
    modeller = {
        "RF": lambda s: RandomForestClassifier(n_estimators=150, n_jobs=N_JOBS, random_state=s),
        "HistGB": lambda s: HistGradientBoostingClassifier(max_iter=200, random_state=s),
    }
    bl = bolmeler7(M, H, seed)
    for b, k, tr, te in bl + arindir7(bl, H):
        if all((b, k, m, o) in tamam for m in modeller for o in setler):
            continue
        rs = np.random.RandomState(seed)
        tr_s = tr if len(tr) <= UST else rs.choice(tr, UST, replace=False)
        te_s = te if len(te) <= UST else rs.choice(te, UST, replace=False)
        sizinti = float(np.isin(H[te], np.unique(H[tr])).mean()) if len(te) else np.nan
        for oad, cols in setler.items():
            Xtr, Xte = np.asarray(X[tr_s])[:, cols], np.asarray(X[te_s])[:, cols]
            for mad, fab in modeller.items():
                if (b, k, mad, oad) in tamam or len(te_s) == 0:
                    continue
                t0 = time.time()
                pb = fab(seed).fit(Xtr, y_bin[tr_s]).predict(Xte)
                pc = fab(seed).fit(Xtr, y_cok[tr_s]).predict(Xte)
                yb, yc, at = y_bin[te_s], y_cok[te_s], aile[te_s]
                r = {"bolme": b, "kat": k, "model": mad, "oznitelik": oad, "seed": seed, "test_n": len(te),
                     "test_sizinti": sizinti,
                     "MCC_ikili": matthews_corrcoef(yb, pb),
                     "F1_makro_cok": f1_score(yc, pc, average="macro"),
                     "MCC_cok": matthews_corrcoef(yc, pc),
                     "yanlis_alarm": float((pb[at == "benign"] == 1).mean()) if (at == "benign").any() else np.nan}
                for a in ("gafgyt", "mirai"):
                    m = at == a
                    r[f"yakalama_{a}"] = float((pb[m] == 1).mean()) if m.any() else np.nan
                    r[f"tur_dogrulugu_{a}"] = float((pc[m] == yc[m]).mean()) if m.any() else np.nan
                for s in siniflar:
                    m = yc == s
                    r[f"geri_{s}"] = float((pc[m] == s).mean()) if m.any() else np.nan
                    if s != "benign":
                        r[f"ikili_yakalama_{s}"] = float((pb[m] == 1).mean()) if m.any() else np.nan
                r["sure_s"] = round(time.time() - t0, 1)
                SON.append(r)
                ck_yaz(ad, SON)
                log(b, k, mad, oad, seed, f"MCC={r['MCC_ikili']:.4f} F1={r['F1_makro_cok']:.4f} leak={sizinti:.3f} ({r['sure_s']}s)")
    log(f"Q seed {seed} bitti: {len(SON)} rows")


if __name__ == "__main__":
    main(int(sys.argv[1]))
