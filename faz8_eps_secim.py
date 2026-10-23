"""Faz 8b: ε seçimi — çok ölçütlü karar desteği (TOPSIS) ve iç içe bir-veri-seti-dışarıda sınama.
Girdi: tables/faz8_profil.csv (şişme), checkpoint/eps_<veri>.npz (faz8_eps.py)
Çıktı: faz8_ciktilar/FAZ8_EPS.md, tables/faz8_eps_adaylar.csv, tables/faz8_eps_icice.csv

Adaylar
  sabit       ε ∈ 10^[-4, 0], 17 değer: τ = pay(d ≤ ε)
  uyarlamalı  c ∈ 10^[-1, 1], 9 değer:  τ = pay(d ≤ c · s), s = bölmenin eğitim içi komşu mesafesi medyanı
              (sıfır olmayanlar; birebir tekrar yoğun veride medyan 0 olacağından)
Ölçütler (aday başına, verilen veri seti kümesinde hesaplanır)
  C1 veri setleri arası Spearman(τ, şişme)                       fayda
  C2 veri seti içi Spearman (veri ortalaması çıkarılmış)          fayda
  C3 küme içinde bir veri seti çıkarılınca en düşük C1            fayda
  C4 τ'nun noktalar arası standart sapması (doyma yok)            fayda
  C5 aynı ailede komşu adaylara göre |ΔC1| ortalaması (plato)     maliyet"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/work/pylibs"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = os.path.expanduser("~/work/IoT_Botnet/faz8_ciktilar")
TAB = f"{OUT}/tables"
MIN_TEST = 200
SABIT = np.logspace(-4, 0, 17)
UYAR = np.logspace(-1, 1, 9)
FAYDA = np.array([1, 1, 1, 1, 0], bool)
AGIRLIK = {"eşit": np.ones(5) / 5, "korelasyon ağırlıklı": np.array([0.35, 0.25, 0.2, 0.1, 0.1])}

prof = pd.read_csv(f"{TAB}/faz8_profil.csv")
G = prof[(prof.n_te_eff >= MIN_TEST) & prof.sisme.notna() & (prof.protokol != "P3")].reset_index(drop=True)
KAYNAK = {"P0": "P0", "P4": "P0", "P1": "P1", "P2": "P2"}   # P4 filtre öncesi P0 ile aynı mesafeler

# her nokta için mesafe dizileri (P2: varlık katlarının listesi)
mes = {}
for v in G.veri.unique():
    z = np.load(f"{OUT}/checkpoint/eps_{v}.npz")
    for p in G[G.veri == v].protokol:
        on = KAYNAK[p] if KAYNAK[p] != "P2" else "P2|"
        anah = sorted({k[:-3] for k in z.files if k.endswith("__d") and (k[:-3] == on or k.startswith(on) and on.endswith("|"))})
        mes[(v, p)] = [(z[a + "__d"], z[a + "__dic"]) for a in anah]


def tau(v, p, aile, deger):
    s = []
    for d, dic in mes[(v, p)]:
        if aile == "sabit":
            e = deger
        else:
            pz = dic[dic > 0]
            e = deger * (np.median(pz) if len(pz) else 0.0)
        s.append((d <= e).mean())
    return float(np.mean(s))


ADAY = [("sabit", e) for e in SABIT] + [("uyarlamalı", c) for c in UYAR]
T = np.array([[tau(v, p, a, e) for a, e in ADAY] for v, p in zip(G.veri, G.protokol)])   # nokta × aday


def sp(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(spearmanr(x, y)[0])


def olcutler(maske):
    g = G[maske]; t = T[maske.values]; y = g.sisme.values; vv = g.veri.values
    C = np.zeros((len(ADAY), 5))
    for j in range(len(ADAY)):
        x = t[:, j]
        C[j, 0] = sp(x, y)
        xd = x - pd.Series(x).groupby(vv).transform("mean").values
        yd = y - pd.Series(y).groupby(vv).transform("mean").values
        C[j, 1] = sp(xd, yd)
        C[j, 2] = min(sp(x[vv != u], y[vv != u]) for u in np.unique(vv))
        C[j, 3] = float(np.std(x))
    for aile in ("sabit", "uyarlamalı"):
        idx = [j for j, (a, _) in enumerate(ADAY) if a == aile]
        for k, j in enumerate(idx):
            kom = [idx[i] for i in (k - 1, k + 1) if 0 <= i < len(idx)]
            C[j, 4] = np.mean([abs(C[j, 0] - C[i, 0]) for i in kom])
    return C


def topsis(C, w):
    R = C / np.where(np.linalg.norm(C, axis=0) == 0, 1, np.linalg.norm(C, axis=0))
    V = R * w
    ideal = np.where(FAYDA, V.max(0), V.min(0)); kotu = np.where(FAYDA, V.min(0), V.max(0))
    dp = np.linalg.norm(V - ideal, axis=1); dn = np.linalg.norm(V - kotu, axis=1)
    return dn / (dp + dn)


def entropi_agirlik(C):
    P = C - C.min(0) + 1e-12
    P = P / P.sum(0)
    E = -(P * np.log(P)).sum(0) / np.log(len(C))
    d = 1 - E
    return d / d.sum()


def ad(j):
    a, e = ADAY[j]
    return f"ε = {e:.3g}" if a == "sabit" else f"ε = {e:.3g} · s"


parca = ["# Faz 8b — ε seçimi: çok ölçütlü karar desteği\n"]
tum = pd.Series(True, index=G.index)
C = olcutler(tum)
skor = {k: topsis(C, w) for k, w in AGIRLIK.items()}
skor["entropi"] = topsis(C, entropi_agirlik(C))
A = pd.DataFrame(C, columns=["C1_arasi", "C2_ici", "C3_en_kotu_LODO", "C4_yayilim", "C5_kararsizlik"])
A.insert(0, "aday", [ad(j) for j in range(len(ADAY))]); A.insert(0, "aile", [a for a, _ in ADAY])
for k, s in skor.items():
    A[f"TOPSIS_{k}"] = s
A.to_csv(f"{TAB}/faz8_eps_adaylar.csv", index=False)
parca.append(f"\n## Tüm veri setleriyle aday tablosu ({len(G)} nokta, {G.veri.nunique()} veri seti)\n\n"
             + A.sort_values("TOPSIS_eşit", ascending=False).round(3).to_markdown(index=False) + "\n")
parca.append("\nEntropi ağırlıkları: " + ", ".join(f"C{i + 1} {w:.2f}" for i, w in enumerate(entropi_agirlik(C))) + "\n")

# iç içe sınama: dışarıdaki veri seti için ε diğerlerinden seçilir
# aynı verinin iki hali (öznitelik seti / yeniden etiketleme) birlikte dışarıda bırakılır
AILE = {"nbaiot_temiz": "nbaiot", "cicids2017_engelen": "cicids2017"}
G["aile"] = G.veri.map(lambda v: AILE.get(v, v))
sat, tah = [], np.full(len(G), np.nan)
for h in sorted(G.aile.unique()):
    m = G.aile != h
    Ch = olcutler(m)
    s = topsis(Ch, AGIRLIK["eşit"])
    j = int(np.argmax(s))
    tah[(~m).values] = T[(~m).values, j]
    sat.append({"disarida": h, "secilen": ad(j), "aile": ADAY[j][0], "TOPSIS": s[j], "ic_C1": Ch[j, 0]})
S = pd.DataFrame(sat)
S.to_csv(f"{TAB}/faz8_eps_icice.csv", index=False)
r_ic, p_ic = spearmanr(tah, G.sisme)
xd = tah - pd.Series(tah).groupby(G.veri.values).transform("mean").values
yd = G.sisme.values - G.groupby("veri").sisme.transform("mean").values
r_ic2, p_ic2 = spearmanr(xd, yd)
parca.append("\n## İç içe sınama (ε dışarıdaki veri ailesi görülmeden seçilir, eşit ağırlık)\n\n" + S.round(3).to_markdown(index=False) + "\n")
parca.append(f"\nDışarıda seçilmiş ε ile birleştirilmiş noktalar: veri setleri arası Spearman = {r_ic:.3f} (p = {p_ic:.3g}); "
             f"veri seti içi Spearman = {r_ic2:.3f} (p = {p_ic2:.3g}); n = {len(G)}.\n")

# karşılaştırma: önceden seçilmiş ε = 0,01
j01 = int(np.argmin([abs(e - 0.01) if a == "sabit" else 9 for a, e in ADAY]))
parca.append(f"\nKarşılaştırma — önceden seçilen ε = 0,01: C1 = {C[j01, 0]:.3f}, C2 = {C[j01, 1]:.3f}.\n")
# veri seti başına en iyi sabit ε (veri seti içi, protokoller arası) ve veri seti özellikleri — ε uyarlama rehberi için
ozel = []
for v, g in G.groupby("veri"):
    idx = g.index.values
    js = [j for j, (a, _) in enumerate(ADAY) if a == "sabit"]
    r = [sp(T[idx, j], g.sisme.values) if len(idx) >= 3 else np.nan for j in js]
    z = np.load(f"{OUT}/checkpoint/eps_{v}.npz")
    dic = np.concatenate([z[k] for k in z.files if k.endswith("__dic")])
    en = np.nanmax(r) if len(idx) >= 3 else np.nan
    iyi = [SABIT[k] for k in range(len(js)) if not np.isnan(r[k]) and r[k] >= en - 1e-9]
    ozel.append({"veri": v, "nokta": len(idx), "en_iyi_r": en,
                 "en_iyi_eps_aralik": f"{min(iyi):.3g}–{max(iyi):.3g}" if iyi else "",
                 "tekrar_payi": g.tekrar_payi.iloc[0], "ic_mesafe_medyan_s": float(np.median(dic[dic > 0])) if (dic > 0).any() else 0.0,
                 "ic_sifir_payi": float((dic == 0).mean()), "kappa": g.kappa.iloc[0]})
parca.append("\n## Veri seti başına en iyi sabit ε (protokoller arası Spearman; nokta sayısı az, yalnızca yönlendirici)\n\n"
             + pd.DataFrame(ozel).round(4).to_markdown(index=False) + "\n")
open(f"{OUT}/FAZ8_EPS.md", "w", encoding="utf-8").write("".join(parca))
print("".join(parca))
