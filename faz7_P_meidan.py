"""Faz 7 P — Reproduction of the original evaluation protocol of Meidan et al. 2018 (N-BaIoT) (torch, GPU).

Protocol in the paper (Literatur Tarama/md/N-BaIoT...md):
  - Each device's benign data is split chronologically into three equal parts: DS_trn, DS_opt, DS_tst (benign).
  - Autoencoder 115-86-58-38-29-38-58-86-115 (75%, 50%, 33%, 25% of the input); stops when DS_opt MSE stops decreasing.
  - tr* = mean(MSE_opt) + std(MSE_opt).
  - ws* = shortest window whose sliding majority vote gives 0% FPR on DS_opt.
  - Test: DS_tst benign + all attack files.
Additions: feature set (115 / 110 without timestamps), preprocessing (z-score / signed log + z-score), 3 seeds;
per-attack-type samples and window TPR; TPR on unique content vectors; share of DS_tst benign rows whose
content is found in DS_trn.
Usage: python3 faz7_P_meidan.py
"""
import time

from faz7_ortak import *  # noqa: F401,F403
from faz7_ortak import ck_oku, ck_yaz, icerik_hash, isaretli_log, log, yukle, zd_ozn
import numpy as np
import torch

DEV = "cuda"


def ae(d, seed):
    torch.manual_seed(seed)
    h = [round(d * f) for f in (0.75, 0.5, 0.33, 0.25)]
    kat, onceki = [], d
    for n in h + h[-2::-1]:
        kat += [torch.nn.Linear(onceki, n), torch.nn.Tanh()]
        onceki = n
    kat.append(torch.nn.Linear(onceki, d))
    return torch.nn.Sequential(*kat).to(DEV)


def mse(net, Z):
    out = []
    with torch.no_grad():
        for i in range(0, len(Z), 65536):
            z = torch.as_tensor(Z[i:i + 65536], device=DEV)
            out.append(((net(z) - z) ** 2).mean(1).cpu().numpy())
    return np.concatenate(out)


def egit(Ztr, Zopt, seed, lr=1e-3, bs=256, max_epok=100, sabir=5):
    net = ae(Ztr.shape[1], seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    X = torch.as_tensor(Ztr, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(seed)
    en_iyi, durum, bekle, epok = np.inf, None, 0, 0
    for epok in range(max_epok):
        p = torch.randperm(len(X), device=DEV, generator=g)
        for i in range(0, len(X), bs):
            b = X[p[i:i + bs]]
            opt.zero_grad(); ((net(b) - b) ** 2).mean().backward(); opt.step()
        v = float(mse(net, Zopt).mean())
        if v < en_iyi - 1e-7:
            en_iyi, durum, bekle = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bekle += 1
            if bekle >= sabir:
                break
    net.load_state_dict(durum)
    return net, epok + 1


def pencere_karar(anomali, ws):
    """Whether the majority is anomalous in every sliding window of length ws (number of windows = n - ws + 1)."""
    if len(anomali) < ws:
        return np.array([anomali.mean() > 0.5])
    c = np.concatenate([[0], np.cumsum(anomali)])
    return (c[ws:] - c[:-ws]) > ws / 2


def ws_bul(anomali_opt, ust=5000):
    for ws in list(range(1, 100)) + list(range(100, ust + 1, 10)):
        if not pencere_karar(anomali_opt, ws).any():
            return ws
    return ust


def main():
    X, M, ozn = yukle()
    Hc = icerik_hash(X, ozn)
    zi = set(zd_ozn(ozn))
    setler = {"hepsi_115": list(range(115)), "zd_yok_110": [j for j in range(115) if j not in zi]}
    SON = ck_oku("P")
    tamam = {r["kayit"] for r in SON}
    for cihaz in sorted(M.cihaz.unique()):
        Mc = M[M.cihaz == cihaz]
        ben = Mc[Mc.aile == "benign"].sort_values("satir_no").index.to_numpy()
        u = len(ben) // 3
        trn, opt_, tst = ben[:u], ben[u:2 * u], ben[2 * u:]
        saldiri = {s: g.sort_values("satir_no").index.to_numpy() for s, g in Mc[Mc.aile != "benign"].groupby("sinif")}
        for oad, cols in setler.items():
            for on in ("z", "log_z"):
                for seed in (42, 7, 123):
                    kayit = f"{cihaz}|{oad}|{on}|{seed}"
                    if kayit in tamam:
                        continue
                    t0 = time.time()
                    A = lambda i: (isaretli_log(np.asarray(X[i])[:, cols]) if on == "log_z" else np.asarray(X[i])[:, cols])
                    Atr = A(trn); m, s = Atr.mean(0), Atr.std(0) + 1e-9
                    Z = lambda i: ((A(i) - m) / s).astype(np.float32)
                    net, epok = egit(Z(trn), Z(opt_), seed)
                    e_opt = mse(net, Z(opt_))
                    tr = e_opt.mean() + e_opt.std()
                    ws = ws_bul(e_opt > tr)
                    e_tst = mse(net, Z(tst))
                    r = {"kayit": kayit, "cihaz": cihaz, "oznitelik": oad, "onisleme": on, "seed": seed,
                         "epok": epok, "tr": float(tr), "ws": ws, "n_trn": len(trn),
                         "FPR_ornek": float((e_tst > tr).mean()),
                         "FPR_pencere": float(pencere_karar(e_tst > tr, ws).mean()),
                         "tst_iyi_trn_icerik_%": 100 * float(np.isin(Hc[tst], Hc[trn]).mean())}
                    tum_a, tum_n = 0, 0
                    for sname, idx in saldiri.items():
                        e = mse(net, Z(idx)); an = e > tr
                        _, ilk = np.unique(Hc[idx], return_index=True)
                        r[f"TPR_ornek_{sname}"] = float(an.mean())
                        r[f"TPR_pencere_{sname}"] = float(pencere_karar(an, ws).mean())
                        r[f"TPR_tekil_{sname}"] = float(an[ilk].mean())
                        r[f"tekil_{sname}"] = int(len(ilk))
                        tum_a += an.sum(); tum_n += len(an)
                    r["TPR_ornek_tum"] = float(tum_a / tum_n)
                    r["sure_s"] = round(time.time() - t0, 1)
                    SON.append(r); ck_yaz("P", SON); tamam.add(kayit)
                    log(kayit, f"ws={ws} FPRp={r['FPR_pencere']:.4f} TPR={r['TPR_ornek_tum']:.4f} "
                        f"tcp={r.get('TPR_ornek_gafgyt:tcp', np.nan):.3f} ({r['sure_s']}s)")
    pd.DataFrame(SON).to_csv(f"{TAB}/P_meidan.csv", index=False)
    log("P bitti")


if __name__ == "__main__":
    main()
