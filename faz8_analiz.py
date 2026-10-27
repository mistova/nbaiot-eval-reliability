"""Faz 8 analysis: profile table, single number (GS), hazard / residual risk / remediability, and validity test.
Input: faz8_ciktilar/tables/faz8_<veri>.csv  ·  Output: faz8_ciktilar/FAZ8_OZET.md and tables/faz8_profil.csv"""
import glob
import os
import sys

sys.path.insert(0, os.path.expanduser("~/work/pylibs"))

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

OUT = os.path.expanduser("~/work/IoT_Botnet/faz8_ciktilar")
TAB = f"{OUT}/tables"
# which feature set the clean reference is taken from (N-BaIoT: the set with timestamp columns removed)
REF_KAYNAK = {"nbaiot": "nbaiot_temiz"}
GRUP = {"nbaiot": "sorunlu", "nbaiot_temiz": "sorunlu", "iot23": "sorunlu", "botiot": "sorunlu", "cicids2017": "sorunlu",
        "cic2023": "kısmen", "kitsune": "kısmen", "medbiot": "temiz aday", "cicids2017_engelen": "temiz aday"}
MIN_TEST = 200          # points with fewer test rows than this are excluded from the validity analysis

parca = []


def bas(baslik, df, ondalik=3, aciklama=""):
    parca.append(f"\n## {baslik}\n")
    if aciklama:
        parca.append(aciklama + "\n\n")
    parca.append("_(tablo yok)_\n" if df is None or len(df) == 0 else df.round(ondalik).to_markdown() + "\n")


D = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{TAB}/faz8_*.csv")) if "profil" not in f], ignore_index=True)
veri = D[D.kayit == "veri"].set_index("veri")
kis = D[D.kayit == "kisayol"].set_index("veri")
B = D[D.protokol.notna()].copy()
B["n_te_eff"] = B.n_te_filtre.fillna(B.n_te_ornek) * (1 - B.gorulmemis_sinif_payi.fillna(0))

# per-protocol profile (entity folds are averaged)
prof = B.groupby(["veri", "protokol"]).agg(
    tau0=("tau0", "mean"), tau_e=("tau_0.01", "mean"), tau_e_001=("tau_0.001", "mean"), tau_e_01=("tau_0.1", "mean"),
    F1=("F1", "mean"), F1_std_tohum=("F1_std", "mean"), F1_kat_std=("F1", "std"), NN1_F1=("NN1_F1", "mean"),
    MCC_ikili=("MCC_ikili", "mean"), n_te_eff=("n_te_eff", "mean"), kat=("kat", "nunique")).reset_index()
prof["kappa"] = prof.veri.map(veri.kappa)
prof["tekrar_payi"] = prof.veri.map(veri.tekrar_payi)
prof["GS"] = prof.kappa * (1 - prof.tau_e)
prof["sigma_k"] = prof.veri.map(kis.HDF_max_F1) / prof.F1
prof["nu"] = prof.NN1_F1 / prof.F1
prof["grup"] = prof.veri.map(GRUP)

# clean reference and inflation (sisme)
ref = {}
for v in prof.veri.unique():
    kaynak = REF_KAYNAK.get(v, v)
    k = prof[(prof.veri == kaynak) & (prof.protokol == "P3")]
    if not len(k) or k.n_te_eff.iloc[0] < MIN_TEST:
        k = prof[(prof.veri == kaynak) & (prof.protokol == "P4")]
    ref[v] = (k.F1.iloc[0], k.protokol.iloc[0]) if len(k) else (np.nan, "")
prof["F1_ref"] = prof.veri.map(lambda v: ref[v][0])
prof["ref_protokol"] = prof.veri.map(lambda v: ref[v][1])
prof["sisme"] = prof.F1 - prof.F1_ref
prof.to_csv(f"{TAB}/faz8_profil.csv", index=False)

goster = ["grup", "tekrar_payi", "kappa", "tau0", "tau_e", "GS", "sigma_k", "F1_std_tohum", "nu", "F1", "F1_ref", "sisme", "n_te_eff"]
bas("Profil — veri seti × protokol (ε = 0,01)", prof.set_index(["veri", "protokol"])[goster],
    3, "P0 rastgele · P1 tekrar grubu ayrık · P2 varlık dışarıda · P3 P2 + yakın ikiz filtresi · P4 P0 + yakın ikiz filtresi. "
       "GS = κ·(1−τε). σk = tek öznitelik F1 / model F1. ν = 1-NN F1 / model F1 (tanı).")

# hazard / residual risk / remediability
rows = []
for v, g in prof.groupby("veri"):
    h = g[g.protokol == "P0"]
    if not len(h):
        continue
    h = h.iloc[0]
    farkinda = g[g.protokol == "P2"] if (g.protokol == "P2").any() else g[g.protokol == "P1"]
    kal = farkinda.iloc[0] if len(farkinda) else None
    rows.append({"veri": v, "grup": GRUP.get(v), "tehlike_tau_e(P0)": h.tau_e, "tehlike_GS(P0)": h.GS,
                 "farkinda_protokol": kal.protokol if kal is not None else "",
                 "kalan_tau_e": kal.tau_e if kal is not None else np.nan, "kalan_GS": kal.GS if kal is not None else np.nan,
                 "giderilebilir_tau_e": h.tau_e - (kal.tau_e if kal is not None else np.nan),
                 "giderilemez_1-kappa": 1 - h.kappa})
bas("Tehlike (P0) — kalan risk (farkında protokol) — giderilebilirlik", pd.DataFrame(rows).set_index("veri"), 3,
    "Farkında protokol: varlık varsa P2 (varlık dışarıda), yoksa P1 (tekrar grubu ayrık). "
    "P3/P4 yakın ikizleri tanım gereği sıfırlar; kalan risk orada yalnızca κ ve kısayoldur.")

# validity: does the metric predict inflation
G = prof[(prof.n_te_eff >= MIN_TEST) & prof.sisme.notna() & ~prof.protokol.isin(["P3"])].copy()
sat = []
for ad, x in (("tau_e", G.tau_e), ("tau0", G.tau0), ("1-GS", 1 - G.GS), ("nu", G.nu), ("sigma_k", G.sigma_k)):
    ok = x.notna() & G.sisme.notna()
    if ok.sum() >= 5:
        r, p = spearmanr(x[ok], G.sisme[ok]); t, pt = kendalltau(x[ok], G.sisme[ok])
        sat.append({"gosterge": ad, "n_nokta": int(ok.sum()), "spearman": r, "p": p, "kendall": t, "p_kendall": pt})
bas("Geçerlilik — göstergelerin şişmeyle sıra korelasyonu (veri seti × protokol noktaları)",
    pd.DataFrame(sat).set_index("gosterge"), 3,
    f"Şişme = F1(nokta) − F1(temiz referans). Referans noktası (P3) analiz dışı; test satırı < {MIN_TEST} olan noktalar dışarıda.")

# independence from the dataset: correlation when each dataset is left out
lodo = []
for v in G.veri.unique():
    g = G[G.veri != v]
    if len(g) >= 5:
        lodo.append({"disarida": v, "spearman_tau_e": spearmanr(g.tau_e, g.sisme)[0],
                     "spearman_1-GS": spearmanr(1 - g.GS, g.sisme)[0]})
bas("Geçerlilik sağlamlığı — bir veri seti dışarıda bırakılınca", pd.DataFrame(lodo).set_index("disarida"), 3)

# ε sensitivity
eps = []
for col, e in (("tau_e_001", 0.001), ("tau_e", 0.01), ("tau_e_01", 0.1)):
    ok = G[col].notna()
    eps.append({"epsilon": e, "spearman": spearmanr(G[col][ok], G.sisme[ok])[0]})
bas("ε duyarlılığı", pd.DataFrame(eps).set_index("epsilon"), 3)

# within dataset (protocol comparison): after subtracting the dataset mean; also without P4 (composition of the small filtered test)
ic = []
for col in ("tau_e_001", "tau_e", "tau_e_01", "tau0"):
    x = G[col] - G.groupby("veri")[col].transform("mean")
    y = G.sisme - G.groupby("veri").sisme.transform("mean")
    H = G[G.protokol != "P4"]
    r, p = spearmanr(x, y); rh, ph = spearmanr(H[col], H.sisme)
    ic.append({"gosterge": col, "veri_ici_spearman": r, "p": p, "P4suz_genel_spearman": rh, "p_P4suz": ph, "n_P4suz": len(H)})
bas("Veri seti içi ve P4'süz korelasyon", pd.DataFrame(ic).set_index("gosterge"), 3,
    "Veri seti içi: her veri setinin ortalaması çıkarılarak protokoller arası değişim karşılaştırılır (tehlike → kalan risk kullanımı).")

bas("Grup ortalamaları (P0, tehlike)", prof[prof.protokol == "P0"].groupby("grup")[["tekrar_payi", "kappa", "tau0", "tau_e", "GS", "sisme"]].mean(), 3)

open(f"{OUT}/FAZ8_OZET.md", "w", encoding="utf-8").write("# Faz 8 — değerlendirme güvenilirliği metriği (otomatik özet)\n" + "".join(parca))
print("written:", f"{OUT}/FAZ8_OZET.md")
