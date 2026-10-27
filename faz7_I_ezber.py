"""Phase 7 I - memorization baseline and near-duplicates (GPU kNN, with the WHOLE training set).

For each split, 20,000 rows from the test set; nearest neighbor over the entire training set (signed log + z-score).
Measures: 1-NN accuracy and macro F1 (performance achievable by memorization alone), exact content duplicate rate,
near-duplicate rates at distance thresholds, per-class 1-NN recall.
Usage: python3 faz7_I_ezber.py
"""
import time

from faz7_ortak import *  # noqa: F401,F403
from faz7_ortak import arindir7, bolmeler7, ck_oku, ck_yaz, icerik_hash, log, oznitelik_setleri, yukle
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef

from faz7_gpu_modeller import Onisleme, en_yakin

N_TEST = 20_000


def main():
    X, M, ozn = yukle()
    H = icerik_hash(X, ozn)
    setler = oznitelik_setleri(ozn)
    y = M.sinif.to_numpy()
    siniflar = np.array(sorted(set(y)))
    yk = np.searchsorted(siniflar, y)
    SON = ck_oku("I")
    tamam = {r["kayit"] for r in SON}
    for seed in (42, 7):
        bl = bolmeler7(M, H, seed)
        for b, k, tr, te in bl + arindir7(bl, H):
            if b == "S2" or len(te) == 0:
                continue
            rs = np.random.RandomState(seed)
            te_s = te if len(te) <= N_TEST else rs.choice(te, N_TEST, replace=False)
            for oad in ("hepsi_115", "zd_yok_110"):
                kayit = f"{b}|{k}|{oad}|{seed}"
                if kayit in tamam:
                    continue
                t0 = time.time()
                cols = setler[oad]
                on = Onisleme().fit(np.asarray(X[rs.choice(tr, min(len(tr), 500_000), replace=False)])[:, cols])
                Ztr = np.concatenate([on(np.asarray(X[tr[i:i + 1_000_000]])[:, cols]) for i in range(0, len(tr), 1_000_000)])
                Zte = on(np.asarray(X[te_s])[:, cols])
                D, I = en_yakin(Ztr, Zte, k=1, parca_te=2048, parca_tr=500_000)
                del Ztr
                d = D[:, 0] / np.sqrt(len(cols))
                p, yt = yk[tr][I[:, 0]], yk[te_s]
                r = {"kayit": kayit, "bolme": b, "kat": k, "oznitelik": oad, "seed": seed,
                     "n_egitim": len(tr), "n_test_ornek": len(te_s),
                     "icerik_tekrar_%": 100 * float(np.isin(H[te_s], np.unique(H[tr])).mean()),
                     "NN1_dogruluk": float((p == yt).mean()), "NN1_F1": f1_score(yt, p, average="macro"),
                     "NN1_MCC": matthews_corrcoef(yt, p),
                     "NN1_ikili_dogruluk": float(((p == np.searchsorted(siniflar, "benign")) ==
                                                  (yt == np.searchsorted(siniflar, "benign"))).mean()),
                     "uzaklik_medyan": float(np.median(d))}
                for e in (1e-6, 1e-3, 1e-2, 1e-1, 0.5):
                    r[f"yakin_tekrar_{e:g}_%"] = 100 * float((d <= e).mean())
                for i, s in enumerate(siniflar):
                    m = yt == i
                    r[f"NN1_geri_{s}"] = float((p[m] == i).mean()) if m.any() else np.nan
                r["sure_s"] = round(time.time() - t0, 1)
                SON.append(r); ck_yaz("I", SON); tamam.add(kayit)
                log(kayit, f"duplicate={r['icerik_tekrar_%']:.1f}% NN1_F1={r['NN1_F1']:.4f} near0.01={r['yakin_tekrar_0.01_%']:.1f}% ({r['sure_s']}s)")
    pd.DataFrame(SON).to_csv(f"{TAB}/I_ezber.csv", index=False)
    log("I done")


if __name__ == "__main__":
    main()
