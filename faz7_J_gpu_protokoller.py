"""Phase 7 J - same splits and feature sets as Q, GPU model families: XGB (cuda), MLP, LR (softmax), kNN-1, kNN-5.
Goal: show that the findings are independent of the model family; kNN-1 also serves as the 'memorization baseline'.
Binary prediction is derived from the multiclass prediction (non-benign -> attack).
Usage: python3 faz7_J_gpu_protokoller.py <tohum> [<tohum> ...]
"""
import sys
import time

from faz7_ortak import *  # noqa: F401,F403
from faz7_ortak import UST, arindir7, bolmeler7, ck_oku, ck_yaz, icerik_hash, log, oznitelik_setleri, yukle
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef

from faz7_gpu_modeller import Onisleme, en_yakin, knn_tahmin, sinir_agi, xgb_gpu

MODELLER = ("XGB", "MLP", "LR", "kNN1", "kNN5")


def main(seed):
    X, M, ozn = yukle()
    H = icerik_hash(X, ozn)
    setler = oznitelik_setleri(ozn)
    y_cok, aile = M.sinif.to_numpy(), M.aile.to_numpy()
    siniflar = np.array(sorted(set(y_cok)))
    yk = np.searchsorted(siniflar, y_cok)
    ben = int(np.searchsorted(siniflar, "benign"))
    ad = f"J_{seed}"
    SON = ck_oku(ad)
    tamam = {(r["bolme"], r["kat"], r["model"], r["oznitelik"]) for r in SON}
    bl = bolmeler7(M, H, seed)
    for b, k, tr, te in bl + arindir7(bl, H):
        if all((b, k, m, o) in tamam for m in MODELLER for o in setler) or len(te) == 0:
            continue
        rs = np.random.RandomState(seed)
        tr_s = tr if len(tr) <= UST else rs.choice(tr, UST, replace=False)
        te_s = te if len(te) <= UST else rs.choice(te, UST, replace=False)
        for oad, cols in setler.items():
            Xtr, Xte = np.asarray(X[tr_s])[:, cols], np.asarray(X[te_s])[:, cols]
            on = Onisleme().fit(Xtr)
            Ztr, Zte = on(Xtr), on(Xte)
            ytr = yk[tr_s]
            # renumbering for classes absent from training (S4/S5)
            var = np.unique(ytr); harita = np.full(len(siniflar), -1); harita[var] = np.arange(len(var))
            ytr2 = harita[ytr]
            komsu = None
            for mad in MODELLER:
                if (b, k, mad, oad) in tamam:
                    continue
                t0 = time.time()
                if mad == "XGB":
                    f = xgb_gpu(Xtr.astype(np.float32), ytr2, seed, len(var)); p2 = f(Xte.astype(np.float32))
                elif mad == "MLP":
                    p2 = sinir_agi(Ztr, ytr2, len(var), (256, 128), 15, seed)(Zte)
                elif mad == "LR":
                    p2 = sinir_agi(Ztr, ytr2, len(var), (), 30, seed, lr=3e-3)(Zte)
                else:
                    if komsu is None:
                        komsu = en_yakin(Ztr, Zte, k=5)
                    p2 = knn_tahmin(komsu[1], ytr2, 1 if mad == "kNN1" else 5)
                pc = var[np.asarray(p2).astype(int)]
                yc, at = yk[te_s], aile[te_s]
                pb, yb = (pc != ben).astype(int), (yc != ben).astype(int)
                r = {"bolme": b, "kat": k, "model": mad, "oznitelik": oad, "seed": seed, "test_n": len(te),
                     "MCC_ikili": matthews_corrcoef(yb, pb),
                     "F1_makro_cok": f1_score(yc, pc, average="macro"),
                     "MCC_cok": matthews_corrcoef(yc, pc),
                     "yanlis_alarm": float((pb[at == "benign"] == 1).mean()) if (at == "benign").any() else np.nan}
                for a in ("gafgyt", "mirai"):
                    m = at == a
                    r[f"yakalama_{a}"] = float((pb[m] == 1).mean()) if m.any() else np.nan
                    r[f"tur_dogrulugu_{a}"] = float((pc[m] == yc[m]).mean()) if m.any() else np.nan
                for i, s in enumerate(siniflar):
                    m = yc == i
                    r[f"geri_{s}"] = float((pc[m] == i).mean()) if m.any() else np.nan
                    if s != "benign":
                        r[f"ikili_yakalama_{s}"] = float((pb[m] == 1).mean()) if m.any() else np.nan
                if mad == "kNN1":
                    d = komsu[0][:, 0] / np.sqrt(len(cols))
                    for e in (0, 1e-4, 1e-2, 1e-1):
                        r[f"yakin_tekrar_{e:g}"] = float((d <= e).mean())
                    r["nn_uzaklik_medyan"] = float(np.median(d))
                r["sure_s"] = round(time.time() - t0, 1)
                SON.append(r); ck_yaz(ad, SON); tamam.add((b, k, mad, oad))
                log(b, k, mad, oad, seed, f"MCC={r['MCC_ikili']:.4f} F1={r['F1_makro_cok']:.4f} ({r['sure_s']}s)")
    log(f"J seed {seed} done: {len(SON)} rows")


if __name__ == "__main__":
    for s in sys.argv[1:]:
        main(int(s))
