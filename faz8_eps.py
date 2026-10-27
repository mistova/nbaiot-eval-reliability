"""Faz 8b: raw nearest-neighbour distances for choosing ε (no model training).
Same splits and same test sample as faz8_metrik; for each split
  d     test → training nearest-neighbour distance / √dimension
  d_ic  training sample → training (excluding itself) nearest-neighbour distance / √dimension (natural density scale of the data)
P3/P4 share the same distances (before filtering) with P2/P0 respectively, so they are not computed separately.
Output: faz8_ciktilar/checkpoint/eps_<veri>.npz"""
import os
import sys
import time

import numpy as np

import faz8_metrik as M
from faz7_ortak import log
from faz7_gpu_modeller import Onisleme, en_yakin

N_IC = 5000      # training sample for the internal scale


def main(ad):
    yol = f"{M.OUT8}/checkpoint/eps_{ad}.npz"
    if os.path.exists(yol):
        log(ad, "eps already exists"); return
    t0 = time.time()
    X, y, v, ozn = M.veri(ad)
    n = len(y)
    H = M.R.ON_HASH if M.R.ON_HASH is not None and len(M.R.ON_HASH) == n else M.R.hashle(np.asarray(X, dtype=np.float64))
    sq = np.sqrt(X.shape[1])
    out = {}
    for kayit, protokol, tr, te, kat in M.bolmeler8(n, H, v):
        if protokol in ("P3", "P4") or len(te) == 0 or len(tr) == 0:
            continue
        rs = np.random.RandomState(0)            # same draw order as faz8_metrik.bolme_degerlendir
        te_s = te if len(te) <= M.N_TE else rs.choice(te, M.N_TE, replace=False)
        tr_nn = tr if len(tr) <= 2_000_000 else rs.choice(tr, 2_000_000, replace=False)
        on = Onisleme().fit(np.asarray(X[rs.choice(tr, min(len(tr), 300_000), replace=False)], dtype=np.float64))
        Z = np.concatenate([on(np.asarray(X[tr_nn[i:i + 500_000]], dtype=np.float64)) for i in range(0, len(tr_nn), 500_000)])
        D, _ = en_yakin(Z, on(np.asarray(X[te_s], dtype=np.float64)), k=1, parca_te=2048, parca_tr=500_000)
        ic = np.random.RandomState(1).choice(len(Z), min(N_IC, len(Z)), replace=False)
        Di, Ii = en_yakin(Z, Z[ic], k=2, parca_te=2048, parca_tr=500_000)
        # excluding itself: if the first neighbour is itself take the second (with an exact twin the first neighbour may be another row, distance still 0)
        d_ic = np.where(Ii[:, 0] == ic, Di[:, 1], Di[:, 0])
        del Z
        out[f"{kayit}__d"] = (D[:, 0] / sq).astype(np.float32)
        out[f"{kayit}__dic"] = (d_ic / sq).astype(np.float32)
        log(ad, "eps", kayit, f"medyan d={np.median(out[kayit + '__d']):.4g}", f"medyan d_ic={np.median(out[kayit + '__dic']):.4g}")
    np.savez_compressed(yol, **out)
    log(ad, "eps bitti", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        try:
            main(a)
        except Exception as e:
            import traceback
            log(a, "HATA", repr(e)); traceback.print_exc()
